#!/bin/bash

# Script to train all four models sequentially using organized config structure
# Usage: bash train_all_models.sh [additional_args...]
# Example: bash train_all_models.sh --trainer.max_epochs=20

echo "🚀 Starting sequential training of all models..."
echo "Additional args: $@"
echo ""

# Function to run training with error handling
train_model() {
    local model_name="$1"
    shift  # Remove the first argument, leaving only additional args
    
    echo "=================================================="
    echo "🔄 Training: $model_name"
    echo "=================================================="
    
    # Run the training command with all additional arguments
    python src/scripts/trainer.py fit \
        --config configs/models/config_base.yaml \
        --config configs/models/$model_name/config.yaml \
        --config configs/models/$model_name/wandb.yaml \
        "$@"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✅ $model_name training completed successfully!"
    else
        echo "❌ $model_name training failed with exit code $exit_code"
        return $exit_code
    fi
    
    echo ""
    return 0
}

# Train all models
start_time=$(date +%s)

echo "📋 Training Queue:"
echo "  1. Baseline Model"
echo "  2. Basic GNN"  
echo "  3. Basic GNN Aug"
echo "  4. Equivariant GNN"
echo ""

# Train each model
train_model "baseline" "$@" || exit 1
train_model "basic_gnn" "$@" || exit 1
train_model "basic_gnn_aug" "$@" || exit 1
train_model "eq_gnn" "$@" || exit 1

# Calculate total time
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

echo "🎉 All models trained successfully!"
echo "⏱️  Total time: ${minutes}m ${seconds}s"