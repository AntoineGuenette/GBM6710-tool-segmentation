# GBM6710-tool-segmentation
Classical computer vision approach for robotic instrument segmentation in minimally invasive surgery. This project reproduces a method combining color filtering, GrabCut refinement, and image features (edges, disparity) weighted by blur level, using the 2017 Robotic Instrument Segmentation Challenge dataset from the da Vinci Xi system.

---

## Requirements for the installation

### Git
Check that git is installed:
```bash
git --version
```
If it is not installed, please follow the official installation instructions for your operating system:
https://git-scm.com/downloads

### Miniconda
Check that Miniconda is installed:
```bash
conda --version
```
If it is not installed, please follow the official installation instructions for your operating system:
https://www.anaconda.com/docs/getting-started/miniconda/install

---

## Installation steps

### Step 1 – Clone the repository
Open a terminal (Command Prompt, PowerShell, or shell) and navigate to the directory where you want to clone the repository.
```bash
cd <path/to/the/repository>
```
Clone the repository:
```bash
git clone https://github.com/AntoineGuenette/GBM6710-tool-segmentation
cd GBM6710-tool-segmentation
```
Verify you're in the correct directory by checking for the required files:
```bash
ls
```
You should see the `data` and `src` folders.

### Step 2 - Setup a Conda environment
Create a dedicated conda environment named **toolseg**:
```bash
conda create -n toolseg python=3.12.12
```
Activate the environment:
```bash
conda activate toolseg
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---
