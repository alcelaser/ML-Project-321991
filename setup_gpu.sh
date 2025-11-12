#!/bin/bash
# GPU Setup Script for TensorFlow in WSL2
# This script sets up the CUDA environment variables

export PATH=/usr/local/cuda-12.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.6

