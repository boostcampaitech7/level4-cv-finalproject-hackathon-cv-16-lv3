# FSDP(Fully Sharded Data Parallel) 사용을 활성화합니다.
export ACCELERATE_USE_FSDP=1 

# CPU RAM 효율적 로딩을 위해 FSDP(Fully Sharded Data Parallel) 설정을 활성화합니다.
export FSDP_CPU_RAM_EFFICIENT_LOADING=1 

# NCCL의 IB(InfiniBand) 사용을 비활성화합니다.
export NCCL_IB_DISABLE=1 


CUDA_VISIBLE_DEVICES=6,7 accelerate launch train.py --cfg-path configs/train_stage1.yaml