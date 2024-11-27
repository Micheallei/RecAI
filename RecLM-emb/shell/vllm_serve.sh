#CUDA_VISIBLE_DEVICES=0,1 
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.90 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 10000 \
  --trust-remote-code

# --enforce-eager \
# --tensor-parallel-size 4 \
# --quantization awq \