# csv-bayes-opt

Train a model from CSV data, check how reliable it is, and get suggestions for the best input setup.

## Install

### Yarn / NPM

```bash
yarn install

# or npm
npm install
```

## Project structure

- `index.js`: Commander routing only (CLI commands and options)
- `commands/`: command handlers (`train`, `validate`, `suggest`)
- `utils/`: data cleaning, model math, scoring, helper functions
- `data/`: default place for sample CSV, saved model, and suggestion logs

## Quick start (uses defaults in `data/`)

```bash
node index.js train
node index.js validate
node index.js suggest
```

## Commands

### 1) Train

```bash
node index.js train --csv data/samples.csv --output data/model.json
```

If you skip options:
- `--csv` defaults to `data/samples.csv`
- `--output` defaults to `data/model.json`

Train steps:
1. Detects number/text columns and prints what it found.
2. Drops rows with too many blanks (> 50% blank).
3. Fills remaining blanks (numbers: median, text: most common value).
4. Adds internal helper columns like `column_was_missing` (used by model only).
5. Finds outliers (> 3 standard deviations) and asks keep/drop.
6. Encodes text to numbers, scales numeric values to 0-1.
7. Trains a Gaussian-process-style model.
8. Validates quality and prints scores.
9. Calculates feature influence ranking.
10. Saves model to JSON.

### 2) Validate

```bash
node index.js validate --model data/model.json
```

If skipped, `--model` defaults to `data/model.json`.

Reprints saved validation and influence ranking without retraining.

### 3) Suggest

```bash
node index.js suggest --model data/model.json --maximize yield
```

If skipped:
- `--model` defaults to `data/model.json`
- `--maximize` is asked interactively

Suggest behavior:
- Generates 2000 random candidates and prints top 3 best options.
- Then asks what inputs you already know.
- It does not ask for the target column you are maximizing.
- Any input left blank is auto-optimized.
- Saves suggestion sessions to `data/suggestions_{timestamp}.json`.

### How target column type is handled

- **Numeric target column**:
  - Optimizer maximizes Expected Improvement.
  - Output shows predicted value with 95% confidence interval.

- **Binary target column** (`TRUE/FALSE` or any 2-value text column):
  - Internally treated as a 1/0 probability target.
  - Output is shown as percentage confidence, for example: `87% likelihood of finishing`.
  - Optimizer maximizes the probability of the most likely binary outcome.

- **Multi-class target column** (3+ text values):
  - Model predicts a probability for every possible class.
  - Output ranks all class probabilities highest to lowest and names the top class first.
  - Example style: `Most likely outcome: TypeB (61%). Alternatives: TypeA (24%), TypeC (15%).`
  - Optimizer searches for inputs that maximize the probability of the single most likely class.

## Plain-English metric guide (good vs bad)

### MAE (Mean Absolute Error)
- Meaning: average prediction mistake size.
- Lower is better.
- `0` means perfect.
- “Good” depends on your unit scale (for example, MAE of `1` is good if values are around `1000`, bad if values are around `2`).

### R² (R-squared)
- Meaning: how well predictions match real values overall.
- Range is usually from `0` to `1` (can be negative if model is very poor).
- Rule of thumb:
  - `>= 0.80`: strong
  - `0.50 to 0.79`: usable / moderate
  - `< 0.50`: weak (warning)

### 95% confidence interval
- Meaning: model gives a range (low to high), not just one number.
- A “good” model has true values inside this range close to 95% of the time.

### Calibration (95%)
- Meaning: the percent of real values that landed inside the model’s 95% range.
- Rule of thumb:
  - `90% to 98%`: healthy
  - `< 85%`: model is overconfident (intervals too narrow)
  - `> 99%`: model is underconfident (intervals too wide)

### Feature importance ranking
- Meaning: which input columns affect predictions the most.
- Higher score = stronger influence.
- Use as directional guidance, not absolute truth.

## Internal columns policy

Internal helper columns (like `*_was_missing`) are:
- used by the model automatically
- never prompted to the user
- never shown in suggestion output
- never written to suggestion session files

When you leave a user input blank during `suggest`, its internal `*_was_missing` flag is silently set to `true`.

## GP implementation details

This project implements Gaussian Process regression directly with [`ml-matrix`](https://www.npmjs.com/package/ml-matrix):
- `Matrix` is used for all matrix operations.
- `CholeskyDecomposition` is used to solve the GP equations.
- No custom hand-written linear algebra solver is used.

## Package-first computation policy

- `danfojs-node` handles dataset cleaning utilities, column statistics, median/mode imputation, one-hot encoding, and min-max normalization.
- `ml-matrix` handles all Gaussian Process matrix operations.
- For large datasets, training automatically falls back to `ml-random-forest` for scalability.
- Custom math is intentionally limited to:
  - Expected Improvement acquisition scoring
  - Multi-category probability reporting text formatting