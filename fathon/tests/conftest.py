"""Test-wide resource limits.

This machine's container is capped at 16 CPUs (cgroup cpu.max) while torch and OpenMP default to
the host's core count, which oversubscribes and makes CPU numerics 5-10x slower. Cap the threads
for the whole test session unless the caller set them explicitly.
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
try:
    import torch
    if 'TORCH_NUM_THREADS' not in os.environ:
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
except ImportError:
    pass
