#!/bin/bash

echo "🔥 Starting Heat Equation 500 Epochs Experiment"
echo "This will take a significant amount of time..."
echo "Results will be saved to experiments/heat_equation_500epochs/"
echo ""

cd /root/autodl-tmp/neuraloperator
python examples/heat_equation_500epochs.py

echo ""
echo "🎉 Heat Equation 500 epochs experiment completed!"
echo "Check experiments/heat_equation_500epochs/ for results"
