#!/usr/bin/env python3
"""
Basic distributed test script for training_hub CI.
Tests tensor creation and all-reduce operations across multiple GPUs.
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize distributed training environment."""
    # Get rank and world size from environment variables set by torchrun
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print(f"Initializing process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Set the CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def test_tensor_creation_and_allreduce():
    """Test basic tensor operations and all-reduce."""
    rank, world_size, local_rank = setup_distributed()
    
    print(f"Rank {rank}: Creating tensor on GPU {local_rank}")
    
    # Create a tensor on the current GPU
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(4, dtype=torch.float32, device=device) * (rank + 1)
    
    print(f"Rank {rank}: Initial tensor: {tensor}")
    
    # Perform all-reduce operation (sum across all processes)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"Rank {rank}: After all-reduce: {tensor}")
    
    # Expected result: sum of (1 + 2 + 3 + 4) = 10 for world_size=4
    expected_sum = sum(range(1, world_size + 1))
    expected_tensor = torch.ones(4, dtype=torch.float32, device=device) * expected_sum
    
    # Verify the result
    if torch.allclose(tensor, expected_tensor):
        print(f"Rank {rank}: ✅ All-reduce test PASSED")
        success = True
    else:
        print(f"Rank {rank}: ❌ All-reduce test FAILED")
        print(f"Rank {rank}: Expected: {expected_tensor}")
        print(f"Rank {rank}: Got: {tensor}")
        success = False
    
    # Synchronize all processes
    dist.barrier()
    
    if rank == 0:
        print(f"Distributed test completed. World size: {world_size}")
        if success:
            print("🎉 All tests PASSED!")
        else:
            print("💥 Some tests FAILED!")
    
    return success


def cleanup_distributed():
    """Clean up distributed environment."""
    dist.destroy_process_group()


def main():
    """Main test function."""
    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("❌ CUDA is not available!")
            return False
        
        print(f"🚀 Starting distributed test with {torch.cuda.device_count()} GPUs")
        
        # Run the test
        success = test_tensor_creation_and_allreduce()
        
        # Clean up
        cleanup_distributed()
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)