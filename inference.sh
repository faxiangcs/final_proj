CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/v4-20260321-104151/checkpoint-46 \
    --val_dataset '/mnt/workspace/final_proj/demo_test.jsonl' \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048

