# MF-NAS
This code is based on the implementation of [DARTS](https://github.com/quark0/darts). 
## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running
To train the model architectures, run this command:
```run
python main.py
```

> To search the model architectures by MF-NAS, set the hyper-parameters in ```config.py``` as such:

| Hyper-parameters   |        Value        |
| ------------------ |-------------------- |
| dataset\_root      |  path\_to\_dataset  |


## Quickly Evaluation with Pre-trained Models

We provide one of the model architecture searched by MF-NAS and the trained weights on CIFAR10 of the model.

To evaluate the performance of pre-trained model on CIFAR10, run:

```eval_cifar_model
python evaluate.py
```

## Results

One of the model achieved by MF-NAS on CIFAR10:

| Model name         | Top 1 Acc | Params (M) |
| ------------------ |---------- |----------- |
| MF-NAS           |   97.52%  |    3.3     |


