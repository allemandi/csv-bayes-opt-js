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

## Pipeline Logic

This tool follows a structured pipeline from raw CSV data to actionable optimization suggestions:

1.  **Data Ingestion & Cleaning**:
    - Loads CSV data and infers column types (numeric or text).
    - Automatically handles missing values by imputing the median for numeric columns and the mode for categorical columns.
    - Identifies and optionally removes 3-sigma numeric outliers.
2.  **Feature Engineering**:
    - Standardizes numeric features using Min-Max scaling.
    - Encodes categorical text features using One-Hot encoding.
3.  **Surrogate Modeling**:
    - Trains a predictive model for *every* column in the dataset, allowing any column to be treated as a target.
    - Uses **Gaussian Process Regression** for small datasets (< 400 rows) for high-fidelity uncertainty estimation.
    - Switches to **Random Forest** for larger datasets to maintain performance.
4.  **Bayesian Optimization**:
    - For numeric targets, it uses the **Expected Improvement (EI)** acquisition function to balance exploration (searching uncertain areas) and exploitation (refining known good areas).
    - For categorical targets, it maximizes the predicted probability of the desired outcome.
5.  **Interactive Suggestions**:
    - Generates thousands of random candidate points.
    - If you provide partial data, the tool "fills in" the missing fields by finding the values that optimize your target.

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
