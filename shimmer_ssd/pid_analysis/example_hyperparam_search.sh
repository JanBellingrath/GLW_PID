#!/bin/bash

# Example usage of the individual discriminator hyperparameter search script
# This script shows how to run a comprehensive hyperparameter search for 
# pretrained encoder models with individual discriminators only.

# Configuration
MODEL_PATH="/path/to/your/model.ckpt"
OUTPUT_DIR="./hyperparam_results"
WANDB_PROJECT="individual-discriminator-search"
WANDB_ENTITY="your-wandb-entity"

# Domain configurations (adjust according to your model)
DOMAIN_CONFIGS='{"domain_name": "v_latents", "module_class": "YourDomainClass"}' \
               '{"domain_name": "t", "module_class": "YourTextDomainClass"}'

# Target configuration for clustering
TARGET_CONFIG="gw_latent"

# Run the hyperparameter search
python hyperparam_search_pretrained_encoders.py \
    --model-path "$MODEL_PATH" \
    --domain-configs "$DOMAIN_CONFIGS" \
    --target-config "$TARGET_CONFIG" \
    --output-dir "$OUTPUT_DIR" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "$WANDB_ENTITY" \
    --sweep-name "individual_discriminator_search_$(date +%Y%m%d_%H%M%S)" \
    --min-clusters 5 \
    --max-clusters 25 \
    --min-layers 2 \
    --max-layers 8 \
    --hidden-dims 32 64 128 256 512 \
    --n-samples 8000 \
    --batch-size 128 \
    --discrim-epochs 30 \
    --count 100 \
    --use-gw-encoded

echo "Hyperparameter search initiated!"
echo "Check progress at: https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT" 