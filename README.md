# UNet
UNet from scratch (pytorch)


> (gpu 장비가 없어서) training 중간에 멈춘 결과입니다.

![inference](https://user-images.githubusercontent.com/96368476/204140350-ba77117f-8bd0-4c5d-91cb-9ffbfc5df0af.png)

## Repository Directory 

``` python 
├── UNet
        ├── datasets
        │     └── cars
        ├── data
        │     ├── __init__.py
        │     └── dataset.py
        ├── option.py
        ├── model.py
        ├── train.py
        ├── inference.py
        └── README.md
```

- `data/__init__.py` : dataset loader
- `data/dataset.py` : data preprocess & get item
- `model.py` : Define block and construct Model
- `option.py` : Environment setting

<br>


## Tutoral

### Clone repo and install depenency

``` python
# Clone this repo and install dependency
git clone https://github.com/inhopp/UNet.git
```

<br>


### train
``` python
python3 train.py
    --device {}(defautl: cpu) \
    --data_name {}(default: cars) \
    --lr {}(default: 0.0001) \
    --n_epoch {}(default: 5) \
    --num_workers {}(default: 2) \
    --batch_size {}(default: 16) \ 
    --eval_batch_size {}(default: 16)
```

### testset inference
```
python3 inference.py
    --device {}(defautl: cpu) \
    --data_name {}(default: cars) \
    --num_workers {}(default: 2) \
    --eval_batch_size {}(default: 16)
```