# Natural Gas Price Prediction using PyTorch

## Abstract

This project evaluates the use of **PyTorch**, an open-source machine learning framework, for forecasting **natural gas prices** based on historical stock market and volume data. The primary goal is to assess the effectiveness of deep learning models—specifically LSTM and Transformer networks—in capturing temporal dependencies in financial time series.

---

## Installation on HPCC

### 1. Load Required Modules
```bash
module load python/3.10-anaconda
```

### 2. Create and Activate Conda Environment
```bash
conda create --name pytorch python=3.10
conda activate pytorch
```

### 3. Install PyTorch
```bash
pip install torch numpy
```

---

## Running the Example Code

### 📁 Directory Structure
- `src/example_lstm.py`: A simple LSTM model that learns to predict a sine wave.
- `scripts/run_lstm_example.sb`: Submission script to run the code using SLURM.

### 🔧 Steps to Run
1. Clone or download the repository.
2. Navigate to the root directory.
3. Submit the job:
```bash
sbatch scripts/run_lstm_example.sb
```
4. After the job completes, check `gas_lstm.out` for output logs.

This minimal example takes less than 5 minutes to run and demonstrates basic usage of PyTorch for time series forecasting on the HPCC.

---

## What is PyTorch?

**PyTorch** is a flexible deep learning framework widely used in science and engineering:
- Time series forecasting
- Natural language processing
- Computer vision
- Reinforcement learning

---

## Classification

PyTorch is a **programming tool/framework**, not middleware or a standalone application. It provides APIs to develop custom ML/DL solutions.

---

## Applications in Science & Engineering

PyTorch helps engineers and scientists:
- Build predictive models
- Analyze temporal relationships in data
- Automate and optimize decision-making

---

## Repository Structure

- `data/`: Input or test datasets
- `models/`: Saved PyTorch models
- `notebooks/`: Jupyter notebooks
- `src/`: Example scripts
- `scripts/`: Submission scripts
- `requirements.txt`: Dependencies
