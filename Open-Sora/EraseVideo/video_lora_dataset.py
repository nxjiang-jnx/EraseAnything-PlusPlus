"""
Video Dataset for Open-Sora LoRA Training
Handles video loading, preprocessing, and batching
Includes text augmentation (shuffle + synonym) for better generalization
"""
import os
import random
from pathlib import Path
from typing import List, Dict

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import decord
decord.bridge.set_bridge('torch')

# NLTK for synonym extraction (matching EraseAnything)
import nltk

# Setup NLTK cache directory
nltk_data_dir = os.path.join(os.path.expanduser("~"), ".cache", "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Download wordnet if needed
if not os.path.exists(nltk_data_dir):
    nltk.download('wordnet', download_dir=nltk_data_dir)
    
# Add to NLTK data path
nltk.data.path.append(nltk_data_dir)

from nltk.corpus import wordnet


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)


class VideoLoraDataset(Dataset):
    def __init__(
        self,
        video_data_root: str,
        instance_prompt: str,
        key_word: str,
        tokenizer_t5=None,  # T5 tokenizer for computing remove_indices
        resolution: int = 256,
        num_frames: int = 33,  # Typical for I2V
        repeats: int = 1,
        center_crop: bool = True,
        shuffle_prob: float = 0.9,
        synonym_prob: float = 0.5,
    ):
        self.video_data_root = Path(video_data_root)
        self.instance_prompt = instance_prompt
        self.key_word = key_word
        self.tokenizer_t5 = tokenizer_t5
        self.resolution = resolution
        self.num_frames = num_frames
        self.repeats = repeats
        self.center_crop = center_crop
        self.shuffle_prob = shuffle_prob
        self.synonym_prob = synonym_prob
        
        # Pre-compute synonyms for the keyword (matching EraseAnything)
        self.synonyms = list(get_synonyms(self.key_word))
        
        # Find all video files
        self.video_data_root = Path(video_data_root)
        if not self.video_data_root.exists():
            raise ValueError(f"Video data root doesn't exist: {video_data_root}")
        
        self.video_files = []
        video_extensions = ['.mp4']
        
        for ext in video_extensions:
            self.video_files.extend(list(self.video_data_root.glob(f"**/*{ext}")))
        
        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {video_data_root}")
        
        print(f"[VideoLoraDataset] Found {len(self.video_files)} video files")
        print(f"[VideoLoraDataset] Resolution: {resolution}, Frames: {num_frames}, Repeats: {repeats}")
        
        # Setup transforms (will be applied per-frame)
        self.resize = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR)
        self.crop = transforms.CenterCrop(resolution) if center_crop else None
    
    def __len__(self):
        return len(self.video_files) * self.repeats
    
    def load_video(self, video_path: str) -> torch.Tensor:
        """
        Load video and sample num_frames
        Returns: tensor of shape (C, T, H, W)
        """
        try:
            vr = decord.VideoReader(str(video_path))
            total_frames = len(vr)
            
            # Sample frames
            if total_frames >= self.num_frames:
                # Uniform sampling
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            else:
                # Repeat last frame if video is too short
                indices = list(range(total_frames))
                indices += [total_frames - 1] * (self.num_frames - total_frames)
            
            frames = vr.get_batch(indices)  # (T, H, W, C)
            frames = frames.float() / 255.0  # Normalize to [0, 1]
            
            # Apply transforms per-frame
            frames_list = []
            for i in range(frames.shape[0]):
                frame = frames[i]  # (H, W, C)
                frame = frame.permute(2, 0, 1)  # (C, H, W)
                
                # Apply resize and crop
                frame = self.resize(frame)
                if self.crop is not None:
                    frame = self.crop(frame)
                
                # Normalize to [-1, 1]
                frame = (frame - 0.5) / 0.5
                
                frames_list.append(frame)
            
            # Stack back to (C, T, H, W)
            frames = torch.stack(frames_list, dim=1)  # (C, T, H, W)
            
            return frames
        
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # Return black video as fallback
            return torch.zeros(3, self.num_frames, self.resolution, self.resolution)
    
    def __getitem__(self, idx):
        idx = idx % len(self.video_files)
        video_path = self.video_files[idx]
        
        # Load video
        video = self.load_video(video_path)
        
        # Create prompt with key_word
        prompt = self.instance_prompt.replace("{}", self.key_word)
        
        # Randomly shuffle word order
        if random.random() < self.shuffle_prob:
            words = prompt.split(" ")
            random.shuffle(words)
            prompt = ' '.join(words)
        
        # Select a synonym for InfoNCE loss and potential replacement
        synonym_word = self.key_word  # Default to original
        if self.key_word and len(self.synonyms) > 0:
            synonym_word = random.choice(self.synonyms)
        
        # Randomly replace keyword with synonym
        current_target_word = self.key_word
        if self.key_word and random.random() < self.synonym_prob and len(self.synonyms) > 0:
            prompt = prompt.replace(self.key_word, synonym_word)
            current_target_word = synonym_word
        
        # Compute remove_indices AFTER augmentation
        # This is critical because token positions have changed!
        remove_indices = None
        if self.key_word and self.tokenizer_t5:
            from utils.find_token import get_word_index
            # Use current_target_word (might be synonym) to find token position
            remove_indices = get_word_index(prompt, current_target_word, self.tokenizer_t5, max_length=256)
        
        return {
            "video": video,
            "prompt": prompt,  # Augmented prompt
            "video_path": str(video_path),
            "remove_indices": remove_indices,  # Token positions in augmented prompt
            "synonym_words": synonym_word,  # For InfoNCE loss (对应 EraseAnything)
            "original_keyword": self.key_word,  # For debugging
        }


def collate_video_fn(examples: List[Dict]) -> Dict:
    """
    Collate function for video batches
    Mimics EraseAnything/lora_dataset.py::collate_data_fn
    """
    videos = torch.stack([example["video"] for example in examples])
    videos = videos.to(memory_format=torch.contiguous_format).float()
    
    prompts = [example["prompt"] for example in examples]
    remove_indices = [example["remove_indices"] for example in examples]
    synonyms = [example["synonym_words"] for example in examples]
    original_keywords = [example["original_keyword"] for example in examples]
    
    batch = {
        "videos": videos,
        "prompts": prompts,
        "remove_indices": remove_indices,
        "synonym_words": synonyms,
        "original_keyword": original_keywords,
    }
    
    return batch

