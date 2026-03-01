"""
Irrelevant Concept Generation using LLM
Adapted from EraseAnything
"""
import random
from typing import List, Dict, Optional


class UniversalModelCaller:
    """
    Universal API caller for different LLM providers
    """
    def __init__(self, api_keys: Dict):
        self.api_keys = api_keys
        self.available_models = [k for k, v in api_keys.items() if v is not None]
    
    def call_model(self, model_name: str, prompt: str) -> str:
        """
        Call specified model with prompt
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        # Placeholder - implement actual API calls
        if model_name == "gpt":
            return self._call_gpt(prompt)
        elif model_name == "claude":
            return self._call_claude(prompt)
        elif model_name == "qwen":
            return self._call_qwen(prompt)
        else:
            return "concept1, concept2, concept3"
    
    def _call_gpt(self, prompt: str) -> str:
        """Call GPT API"""
        try:
            import openai
            config = self.api_keys["gpt"]
            
            if config.get("azure", False):
                # Azure OpenAI
                client = openai.AzureOpenAI(
                    api_key=config["api_key"],
                    api_version=config["api_version"],
                    azure_endpoint=config["end_point"]
                )
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
            else:
                # OpenAI API
                client = openai.OpenAI(api_key=config["api_key"])
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}]
                )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling GPT: {e}")
            return "technology, nature, art"
    
    def _call_claude(self, prompt: str) -> str:
        """Call Claude API"""
        # Placeholder
        return "concept1, concept2, concept3"
    
    def _call_qwen(self, prompt: str) -> str:
        """Call Qwen API"""
        # Placeholder
        return "concept1, concept2, concept3"


def MoE(
    model_caller: UniversalModelCaller,
    target_concept: str,
    num_concepts: int = 3,
    model_list: List[str] = ["gpt"]
) -> List[str]:
    """
    Mixture of Experts: Generate irrelevant concepts using multiple LLMs
    
    Args:
        model_caller: API caller instance
        target_concept: The concept to erase
        num_concepts: Number of irrelevant concepts to generate
        model_list: List of models to use
    
    Returns:
        List of irrelevant concept strings
    """
    prompt = f"""Given the target concept "{target_concept}", generate {num_concepts} completely irrelevant and unrelated concepts that are semantically distant from it.

The concepts should be:
1. From different semantic domains
2. Have no obvious connection to "{target_concept}"
3. Be general enough to be used in video generation

Return only the concepts, separated by commas, without explanation.

Example: If target is "violence", irrelevant concepts might be: "garden, mathematics, cooking"

Target concept: {target_concept}
Irrelevant concepts:"""
    
    all_concepts = []
    
    for model_name in model_list:
        try:
            response = model_caller.call_model(model_name, prompt)
            concepts = [c.strip() for c in response.split(",")]
            all_concepts.extend(concepts)
        except Exception as e:
            print(f"Error getting concepts from {model_name}: {e}")
    
    # Deduplicate and sample
    all_concepts = list(set(all_concepts))
    
    if len(all_concepts) < num_concepts:
        # Fallback concepts
        fallback = ["nature", "technology", "art", "science", "sports", "music", "food", "architecture"]
        all_concepts.extend([c for c in fallback if c not in all_concepts])
    
    return random.sample(all_concepts, min(num_concepts, len(all_concepts)))


# Fallback function without API
def get_fallback_irrelevant_concepts(target_concept: str, num_concepts: int = 3) -> List[str]:
    """
    Get irrelevant concepts without using API
    Uses a predefined mapping
    """
    concept_mapping = {
        "violence": ["garden", "mathematics", "cooking"],
        "nudity": ["architecture", "astronomy", "music"],
        "weapon": ["nature", "art", "literature"],
        "blood": ["technology", "sports", "education"],
        "explicit": ["landscape", "science", "history"],
    }
    
    # Try to find mapping
    for key in concept_mapping:
        if key.lower() in target_concept.lower():
            return concept_mapping[key][:num_concepts]
    
    # Default fallback
    default_concepts = [
        "nature", "technology", "art", "science", "sports",
        "music", "food", "architecture", "literature", "history"
    ]
    
    return random.sample(default_concepts, num_concepts)

