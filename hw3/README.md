# PPO Implementation and Comparison

This project implements the **Proximal Policy Optimization (PPO)** reinforcement learning algorithm from scratch and compares it with baseline methods, namely **Actor-Critic (AC)** and **Advantage Actor-Critic (A2C)**. The code is written in Python using PyTorch and OpenAI Gym, and includes visualizations, experimental configurations.

---

## 🚀 Features

- Complete PPO agent implementation using PyTorch
- Integration of:
  - Reward clipping
  - Entropy regularization
- Baseline comparisons with:
  - Actor-Critic (AC)
  - Advantage Actor-Critic (A2C)
- Evaluation over:
  - Multiple random seeds
  - Different environment steps (e.g., \(10^5\) and \(10^6\))
- Real-time CSV logging for experiment reproducibility
- Smoothed reward visualizations using Matplotlib

---

## 📊 Results Summary
please complete this
---

## 📦 Installation

### Option 1: Install via `requirements.txt`

```bash
pip install -r requirements.txt
```
### Option 2: Manual Installation
#### PyTorch (choose one based on your environment)
+ For GPU (CUDA 11.7)
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
+ For CPU
```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
```
#### Other packages
```bash
pip install gym==0.26.2
pip install numpy==1.23.5
pip install matplotlib==3.7.0
pip install pandas==1.5.3
```
⚠️ If you're using Colab or an environment with pre-installed libraries, check your current PyTorch and TensorFlow versions using:
```bash
pip show torch torchvision torchaudio tensorflow
```
### 🧪 How to Run
You can open and run the notebooks either:

+ In Google Colab (especially for TensorFlow notebooks)

+ In your local Jupyter Notebook/Lab environment

## 📁 File Structure

1. RandomBaselineCartPole.csv : Given in the first assignment. This is baseline DQN results given by the TAs.
2. baselinegraphing.py : The csv baseline is smoothed and plotted to get visualisation.
3. AC&A2C_1E5.py : Implementing AC and A2C with number of environment steps set to 10^5.
4. AC&A2C_1E6.py :  Implementing AC and A2C with number of environment steps set to 10^6.
5. ac_a2c_results_1e6.csv:  As there are 10^6 environment steps per run and there are 5 runs results are saved (for both AC and A2C) in csv file to be plotted later.
6. PPO_1e5.py: Implementing PPO with number of environment steps set to 10^5.
7. PPO_1e6.py: Implementing PPO with number of environment steps set to 10^6.
8. PPO_results_1e6.csv : As there are 10^6 environment steps per run and there are 5 runs results are saved in csv file to be plotted later.
9. PPO_with_SGD_1e5.py :  Implementing PPO WITH SGD (mini batching) and number of environment steps set to 10^5.
10. PPO_with_SGD_1e6.py :  Implementing PPO WITH SGD (mini batching) and number of environment steps set to 10^6.
11. ppo_sgd_results_1e6.csv : As there are 10^6 environment steps per run and there are 5 runs results are saved in csv file to be plotted later.This is PPO WITH sgd or mini batching.
15. average_comparison_graphs_all.py :  comparision graphs. every graph contains 2 models. for e.g. ppo vs ppo with sgd, AC vs PPO, A2C vs PPO (all 10^6 model pairs).
16. average_comparison_graphs_all2.py : comparision graphs. multigraph for 10^5 models and some 10^6 models
17. requirements.txt: The dependencies and libraries used in this project.

