#!/bin/bash

# Array of model IDs to train
MODEL_IDS=(
    "mistralai/Mistral-Nemo-Instruct-2407"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting training runs for ${#MODEL_IDS[@]} models...${NC}"
echo "Models to train: ${MODEL_IDS[*]}"
echo "----------------------------------------"

# Track start time
SCRIPT_START_TIME=$(date +%s)

# Loop through each model ID
for i in "${!MODEL_IDS[@]}"; do
    MODEL_ID="${MODEL_IDS[$i]}"
    
    echo -e "\n${YELLOW}[$(($i + 1))/${#MODEL_IDS[@]}] Starting training for: ${MODEL_ID}${NC}"
    echo "Time: $(date)"
    
    # Track individual model start time
    MODEL_START_TIME=$(date +%s)
    
    # Run the training script
    if python confuse_run.py "$MODEL_ID"; then
        MODEL_END_TIME=$(date +%s)
        MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
        echo -e "${GREEN}✓ Successfully completed training for: ${MODEL_ID}${NC}"
        echo -e "Duration: ${MODEL_DURATION} seconds ($(date -u -d @${MODEL_DURATION} +%H:%M:%S))"
    else
        echo -e "${RED}✗ Failed to train model: ${MODEL_ID}${NC}"
        echo "Check the logs above for error details."
        # Uncomment the next line if you want to stop on first failure
        # exit 1
    fi
    
    echo "----------------------------------------"
done

# Calculate total script duration
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))

echo -e "\n${GREEN}All training runs completed!${NC}"
echo "Total duration: ${TOTAL_DURATION} seconds ($(date -u -d @${TOTAL_DURATION} +%H:%M:%S))"
echo "Finished at: $(date)"
