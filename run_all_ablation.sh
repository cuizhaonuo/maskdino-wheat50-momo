#!/bin/bash

echo "======================================"
echo "Running ALL ablation experiments"
echo "======================================"

bash train_ablation_final_maskdino.sh

echo ""
echo "--------------------------------------"
echo "First batch finished"
echo "--------------------------------------"

bash train_ablation_final_noweight_maskdino.sh

echo ""
echo "======================================"
echo "ALL ablation experiments finished"
echo "======================================"