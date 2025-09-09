# REINFORCE and basic Actor-Critic Methods
This project explores and compares key reinforcement learning algorithms — **REINFORCE**, **Actor-Critic (AC)**, and **Advantage Actor-Critic (A2C)** — implemented using both **PyTorch** and **TensorFlow**.

## 🧠 Algorithms Implemented

### ✅ PyTorch Versions
- `REINFORCE`
- `Actor-Critic (AC)`
- `Advantage Actor-Critic (A2C)`
- Multiple config-based variations and exploration notebooks

### ✅ TensorFlow Versions
- `REINFORCE`
- `Actor-Critic (AC)`
- `Advantage Actor-Critic (A2C)`

## 📁 File Structure

### PyTorch Notebooks
- `PyTorch reinforce_algorithm_exploration.ipynb`: Experimentation with REINFORCE
- `PyTorch configs_1/2/3_reinforce_algorithm.ipynb`: Variants using different hyperparameters/configs
- `PyTorch AC vs A2C code.ipynb`: Comparative implementation of AC vs A2C

### TensorFlow Notebooks
- `tensorflow reinforce, AC, A2C.ipynb`: Combined implementation and comparison
- `tensorflow AC vs Reinforce.ipynb`: Focused comparison of AC and REINFORCE

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
pip install matplotlib
pip install tensorflow==2.11.0  # Optional: Adjust version as needed
```
⚠️ If you're using Colab or an environment with pre-installed libraries, check your current PyTorch and TensorFlow versions using:
```bash
pip show torch torchvision torchaudio tensorflow
```
### 🧪 How to Run
You can open and run the notebooks either:

+ In Google Colab (especially for TensorFlow notebooks)

+ In your local Jupyter Notebook/Lab environment
### 📊 Results & Comparisons
The notebooks include:

+ Training curves (reward vs. episode)
+ Performance comparison across parameter settings and algorithms 

We strongly suggest you look at our **full code directory**, including all PyTorch and TensorFlow implementations, training logs, and result plots, available in the shared Google Drive folder:

👉 [Access the full code and results on Google Drive](https://drive.google.com/drive/folders/1YxT02UzoZL7-MM9bMDIsczBGtdk7YeF-?usp=sharing)
