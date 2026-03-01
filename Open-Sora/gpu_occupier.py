#!/usr/bin/env python3
"""
GPU内存占用脚本
用于占用指定GPU的内存，使缓存居高不下
"""

import torch
import time
import argparse
import sys
import signal
import os

class GPUOccupier:
    def __init__(self, gpu_id=0, memory_fraction=0.95):
        """
        初始化GPU占用器
        
        Args:
            gpu_id: GPU设备ID
            memory_fraction: 占用GPU内存的比例 (0.0-1.0)
        """
        self.gpu_id = gpu_id
        self.memory_fraction = memory_fraction
        self.tensors = []
        self.running = True
        
        # 检查CUDA是否可用
        if not torch.cuda.is_available():
            print("错误: CUDA不可用")
            sys.exit(1)
            
        # 检查指定的GPU是否存在
        if gpu_id >= torch.cuda.device_count():
            print(f"错误: GPU {gpu_id} 不存在，可用GPU数量: {torch.cuda.device_count()}")
            sys.exit(1)
            
        # 设置当前设备
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')
        
        print(f"正在占用GPU {gpu_id}...")
        print(f"GPU名称: {torch.cuda.get_device_name(gpu_id)}")
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """处理中断信号"""
        print(f"\n收到信号 {signum}，正在清理资源...")
        self.cleanup()
        sys.exit(0)
        
    def get_gpu_memory_info(self):
        """获取GPU内存信息"""
        total_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(self.gpu_id)
        cached_memory = torch.cuda.memory_reserved(self.gpu_id)
        
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': total_memory - cached_memory
        }
        
    def occupy_memory(self):
        """占用GPU内存"""
        memory_info = self.get_gpu_memory_info()
        target_memory = int(memory_info['total'] * self.memory_fraction)
        
        print(f"总内存: {memory_info['total'] / 1024**3:.2f} GB")
        print(f"目标占用: {target_memory / 1024**3:.2f} GB ({self.memory_fraction*100:.1f}%)")
        
        # 计算需要分配的内存块大小
        block_size = 100 * 1024 * 1024  # 100MB per block
        allocated = 0
        
        try:
            while allocated < target_memory and self.running:
                # 计算当前块的大小
                remaining = target_memory - allocated
                current_block_size = min(block_size, remaining)
                
                # 计算tensor的大小 (float32 = 4 bytes)
                tensor_size = current_block_size // 4
                
                # 创建tensor并移动到GPU
                tensor = torch.randn(tensor_size, device=self.device)
                self.tensors.append(tensor)
                
                allocated += current_block_size
                
                # 显示进度
                progress = (allocated / target_memory) * 100
                print(f"\r已分配: {allocated / 1024**3:.2f} GB ({progress:.1f}%)", end='', flush=True)
                
        except torch.cuda.OutOfMemoryError:
            print(f"\n内存分配完成，已达到GPU内存限制")
            
        print(f"\n成功占用GPU {self.gpu_id}内存")
        
    def keep_alive(self):
        """保持内存占用，防止被释放"""
        print("正在保持内存占用状态...")
        print("按 Ctrl+C 退出")
        
        while self.running:
            # 定期访问tensor以防止被垃圾回收
            for tensor in self.tensors:
                if self.running:
                    # 简单的计算来保持tensor活跃
                    _ = tensor.sum()
                    
            # 显示当前内存状态
            memory_info = self.get_gpu_memory_info()
            print(f"\rGPU {self.gpu_id} 内存状态 - "
                  f"已分配: {memory_info['allocated']/1024**3:.2f}GB, "
                  f"缓存: {memory_info['cached']/1024**3:.2f}GB, "
                  f"可用: {memory_info['free']/1024**3:.2f}GB", 
                  end='', flush=True)
            
            time.sleep(1)
            
    def cleanup(self):
        """清理资源"""
        print("\n正在清理GPU内存...")
        self.running = False
        
        # 删除所有tensor
        for tensor in self.tensors:
            del tensor
            
        self.tensors.clear()
        
        # 清理CUDA缓存
        torch.cuda.empty_cache()
        
        print("GPU内存已释放")
        
    def run(self):
        """运行GPU占用器"""
        try:
            self.occupy_memory()
            self.keep_alive()
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='GPU内存占用脚本')
    parser.add_argument('--gpu', type=int, default=0, help='GPU设备ID (默认: 0)')
    parser.add_argument('--memory-fraction', type=float, default=0.95, 
                       help='占用GPU内存的比例 (默认: 0.95)')
    
    args = parser.parse_args()
    
    # 验证参数
    if not 0 <= args.memory_fraction <= 1:
        print("错误: memory-fraction 必须在 0.0 到 1.0 之间")
        sys.exit(1)
        
    # 创建并运行GPU占用器
    occupier = GPUOccupier(args.gpu, args.memory_fraction)
    occupier.run()

if __name__ == "__main__":
    main()
