##  CIPGeo: a Confidence-Interval Prediction framework for Geolocation
Towards Trustworthy IP Geolocation Services: A Conformal Interval Prediction Framework

```bash
CIPGeo
├── assets  # log
│   ├── interval_width_Los_Angeles_10.0.json
│   ├── interval_width_New_York_10.0.json
│   ├── interval_width_Shanghai_0.1.json
├── datasets  
│   ├── Los_Angeles
│   ├── New_York
│   └── Shanghai
├── lib  
│   ├── dataset.py
│   ├── model.py
│   ├── preprocess.py
│   └── utils.py
└── main.py
```


**Environment：**

```bash
conda create -n exgnn python=3.9
conda activate exgnn
pip install -r requirements.txt
```

**Datasets（The same as the TrustGeo datasets）：**

https://github.com/ICDM-UESTC/TrustGeo

Download it and place it in the "datasets" folder.


**run：**
# Open the "CIPGeo" folder
```bash
cd CIPGeo
```
# data preprocess (executing IP clustering).
```bash
python main.py --norm_x --dataset New_York --lambda_len 10 --dim_in 32 --lr 5e-3 --epochs 150 --seed 2025 --model_name QRGATGeo
python main.py --norm_x --dataset Los_Angeles --lambda_len 10 --dim_in 32 --lr 1e-2 --epochs 150 --seed 2025 --model_name QRGATGeo
python main.py --norm_x --dataset Shanghai --lambda_len 0.1 --dim_in 53 --lr 1e-2 --epochs 150 --seed 2025 --model_name QRGATGeo
```
