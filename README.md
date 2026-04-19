# DS 340W M-Help Baseline Project

This project studies AI detection of mental health help-seeking signals from
Reddit posts. The task is binary text classification:

- `help = 1`: the post shows help-seeking behavior
- `help = 0`: the post does not show help-seeking behavior

The project uses only single Reddit posts. It does not use user-level modeling,
multiple posts per user, or temporal/longitudinal modeling.

## Data

The project uses three CSV files in `data/`:

- `train_data.csv`: 1142 rows
- `val_data.csv`: 360 rows
- `test_data.csv`: 369 rows

The original dataset includes `ID`, `Text`, `help`, `Cause`, and
`MH-condition`. This project only uses `Text` and `help`.

## Code Files

- `notebooks/exploration.ipynb`: original exploratory baseline notebook
- `notebooks/enhanced_experiment.ipynb`: enhanced notebook version for presentation
- `enhanced_mhelp_experiment.py`: enhanced comparison experiment for the DS 340W novelty assignment
- `report_novelty_section.md`: draft text that can be added to the written report
- `run_project.sh`: one-command runner for Mac/Linux
- `run_project.bat`: one-command runner for Windows

## Enhanced Experiment

The enhanced experiment compares several methods:

- majority-class baseline
- original word TF-IDF with logistic regression
- balanced logistic regression
- word TF-IDF with linear SVM
- complement Naive Bayes
- character n-gram TF-IDF with linear SVM

The character n-gram TF-IDF model is the main novelty block. It is designed to
capture informal spelling, punctuation, and short emotional expressions common
in Reddit posts.

## Running the Experiment

After downloading or cloning the GitHub repository, open a terminal in the main project folder.

On Mac/Linux, run:

```bash
bash run_project.sh
```

On Windows, run:

```bat
run_project.bat
```

The runner creates a virtual environment, installs the required packages, runs the enhanced experiment, prints the comparison table, and saves result files in `results/`.

Expected main test results:

| Model | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|
| Original Word TF-IDF + Logistic Regression | 0.604 | 0.539 | 0.563 |
| Character n-gram TF-IDF + Linear SVM | 0.602 | 0.583 | 0.595 |
