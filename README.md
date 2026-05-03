# csv-bayes-opt

A Node.js CLI tool for training models and generating optimization suggestions from CSV data using Gaussian Processes and Random Forests.

## Installation

```bash
yarn install
```

## Quick Start

```bash
# Train a model from data/samples.csv
node index.js train

# Validate the saved model
node index.js validate

# Get optimization suggestions
node index.js suggest
```

## Commands

### `train`
Trains a model on your CSV data.
- **Data Cleaning**: Automatically handles missing values and outliers.
- **Model Selection**: Uses Gaussian Process Regression for small datasets and automatically switches to **Random Forest** when the dataset reaches **400 rows** for better scalability.
- **Evaluation**: Provides MAE, R², and feature importance rankings.

### `validate`
Prints the stored validation report and feature influence ranking from a saved model JSON.

### `suggest`
Interactive optimization engine.
- **Targeting**: Choose any numeric or categorical column to maximize.
- **Automation**: Generates top 3 automatic suggestions.
- **Interaction**: Allows fixing specific inputs to see how the model optimizes the remaining variables.

## Key Metrics

- **MAE**: Average prediction error (lower is better).
- **R²**: Goodness of fit (0 to 1; >0.7 is strong, <0.5 is weak).
- **95% Calibration**: Percentage of real values falling within the predicted 95% confidence interval.

## Development

```bash
# Run linting
yarn lint

# Run unit tests
yarn test
```
