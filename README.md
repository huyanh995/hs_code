# HS Code Classification (Vietnamese)

The implementation of the paper

```
Attentive RNN for HS Code Hierarchy Classification on Vietnamese Goods Declaration
Nguyen Thanh Binh, Huy Anh Nguyen, Pham Ngoc Linh, Nguyen Linh Giang, Tran Ngoc Thang
Proceedings of the International Conference on Intelligent Systems and Networks (ICISN), 2021
```
Link to paper: https://link.springer.com/chapter/10.1007/978-981-16-2094-2_37

## Installation (Conda)
```
conda create -n hscode python=3.8.5 -y
```
## Traning
```
python train.py --data train.csv --epochs 200
```

## Testing (Demo)
```
python test.py --data sample.csv
```
## Data
Provide upon request. Contact [thang.tranngoc@hust.edu.vn](mailto:thang.tranngoc@hust.edu.vn)
## Citation
If you find the code useful, please cite:

```
@InProceedings{10.1007/978-981-16-2094-2_37,
author= {Binh, Nguyen Thanh and Nguyen, Huy Anh and Linh, Pham Ngoc and Giang, Nguyen Linh and Thang, Tran Ngoc},
title= {Attentive RNN for HS Code Hierarchy Classification on Vietnamese Goods Declaration},
booktitle= {Intelligent Systems and Networks},
year= {2021},
publisher= {Springer Singapore},
address= {Singapore},
pages= {298--304},
isbn= {978-981-16-2094-2}
}
```
