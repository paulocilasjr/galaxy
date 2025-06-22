# Tabular Learner Tools

This repository contains two machine learning tools for working with tabular data in the Gleam framework:

## 1. Tabular Learner

A comprehensive tool for training and evaluating multiple machine learning models on tabular datasets.

### Features:
- Supports both classification and regression tasks
- Automatically compares multiple algorithms to find the best model
- Extensive customization options:
  - Data normalization
  - Feature selection
  - Cross-validation
  - Outlier removal
  - Multicollinearity handling
  - Polynomial feature generation
  - Class imbalance correction
- Outputs detailed HTML reports with performance metrics and visualizations
- Saves the best model for later use

## 2. PyCaret Predictor/Evaluator

A companion tool for making predictions and evaluating trained models on new data.

### Features:
- Works with models trained by Tabular Learner
- Supports both classification and regression tasks
- Generates predictions on new data
- Creates evaluation reports when target values are provided
- Outputs predictions in CSV format

## Workflow

These tools are designed to work together:
1. Use **Tabular Learner** to train and find the best model for your dataset
2. Use **PyCaret Predictor/Evaluator** to apply your trained model to new data

Both tools are powered by [PyCaret](https://pycaret.org/), an open-source machine learning library that automates the ML workflow.