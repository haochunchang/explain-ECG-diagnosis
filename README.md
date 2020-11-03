<div align="center">    
 
# ECG Diagnosis

![CI testing](https://github.com/haochunchang/ECG-diagnosis/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>
 
## Description   
Scripts and modules for training, testing and explaining deep neural networks for ECG automatic classification.

The data is from [The PTB Diagnostic ECG Database](https://archive.physionet.org/physiobank/database/ptbdb/).


## How to run
First, install dependencies   
```bash
# clone project   
git clone https://github.com/haochunchang/ECG-diagnosis.git

# install project
cd ECG-diagnosis
pip install -e .
pip install -r requirements.txt

# run main script
python main.py --help
```
