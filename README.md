<div align="center">    
 
# Explaining ECG Diagnosis

![CI testing](https://github.com/haochunchang/explain-ECG-diagnosis/workflows/CI%20testing/badge.svg?branch=master&event=push)

</div>

## How to run
```bash
# clone project   
git clone git@github.com:haochunchang/explain-ECG-diagnosis.git
cd ECG-diagnosis

# install project
pip install -e .
pip install -r requirements.txt

# run main script
python main.py --help
```

<div style="padding-top: 0.5em">

## Description   
Scripts and modules for training, testing and explaining deep neural networks for classifying **reason of admission** from **ECG signals**.

</div>

### Data
The data is from [The PTB Diagnostic ECG Database](https://archive.physionet.org/physiobank/database/ptbdb/).

> The database contains 549 records from 290 subjects. Each subject is represented by 1 ~ 5 records

> Each record includes **15** simultaneously measured signals:
> * The conventional 12 leads (i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6)
> * The 3 Frank lead ECGs (vx, vy, vz).

> Each signal is digitized at 1000 samples per second.


### Current setting

- Preprocessing
    * Splits records into a fixed chunk size, treating each chunk as individual sample.
- Model
    * Use 1D convolutional filters and fully-connected layers to learn signal features.
- Evaluation
    * Using Accuracy, F1 score and confusion matrix on testing dataset.
- Interpretation
    * Modified Gradient-weighted Class Activation Mapping (Grad-CAM)
