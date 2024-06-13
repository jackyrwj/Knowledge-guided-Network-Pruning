python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=2345 \
    exp1_pretrain_model_cross_subject2.py