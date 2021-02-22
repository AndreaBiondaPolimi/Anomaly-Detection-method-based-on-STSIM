# Anomaly Detection based on STSIM as Feature Extraction
## Description
This repository contains the Data-Driven approach of the Politecnico di Milano Master Thesis about Anomaly Detection: \title

## Usage
Execute the software 
* TYPE: type of anomaly metrics to use, one of: 'mahalanobis' or 'KDE'
* FILE: configuration file path

```sh
cd Anomaly_Detection_STSIM
python Main.py -t TYPE -f FILE 
```
It is possible to configure the implementation parameters in [Parameters.ini](Configuration/Parameters.ini).


## Installation
Clone and install: 
```sh
git clone https://github.com/AndreaBiondaPolimi/Anomaly_Detection_STSIM.git
cd Anomaly_Detection_STSIM
pip install -r requirements.txt
```

## Requirements
* opencv-python==4.1.1.26
* numpy>=1.16.4
* scipy==1.4.1
* matplotlib==3.1.1
* tensorflow-gpu==2.1.0
* scikit-image>=0.16.2
* scikit-learn==0.21.3
* albumentations==0.4.5