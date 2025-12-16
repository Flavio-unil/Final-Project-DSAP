# Final Project DSAP — Can Economic Sentiment Predict Market Movements?

This project studies whether Google search interest for economic-related keywords can help predict next week's S&P 500 return.

We build sentiment indices from Google Trends weekly search intensity and evaluate simple predictive models using an out-of-sample split.

## Project structure

- `main.py`: end-to-end pipeline (load data → build sentiment features → train/test split → model comparison → save outputs)
- `datasets/clean_data/merged_weekly.csv`: merged weekly dataset used by the pipeline
- `analysis/`: Jupyter notebooks used during exploration and feature construction
- `outputs/`: generated results and figures

## Data

The pipeline expects the following file to exist:

- `datasets/clean_data/merged_weekly.csv`

This dataset contains weekly Google Trends series (keywords) and market variables, and is used to build sentiment features.

## How to run

From the project root directory:

```bash
python main.py
