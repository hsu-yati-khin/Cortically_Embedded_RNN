# CERNN Project
This project develops Cortically-Embedded Recurrent Neural Networks (CERNNs) for comparing brain network connectivity for cross-species including Human and Mouse, implemented through gating mechanisms in a Leaky RNN. 


## Installation

You may need to install PyTorch separately with the correct CUDA version for your system.  
  Visit the official PyTorch installation page for detailed instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Training models requires an account on Weights & Biases (wandb).  
  You can create a free account here: [https://wandb.ai/site](https://wandb.ai/site)



To install the required Python packages...

Run the following command or use `requirement.txt` file below:

```bash
pip install hydra-core matplotlib numpy omegaconf pytorch_lightning PyYAML scipy seaborn six torch wandb
```

To upgrade all packages to their latest versions, run:

```bash
pip install --upgrade hydra-core matplotlib numpy omegaconf pytorch_lightning PyYAML scipy seaborn six torch wandb
```

Using the `requirements.txt` 

```setup
pip install -r requirements.txt
```


## Datasets

### Mouse Dataset

To obtain the mouse datasets in CSV format, please first run the notebook file named "mouse_dataset_cleaning.ipynb", since the datasets are too large to be uploaded to the Git repository.


## 📝 Setup Instructions

Before training the model:
1. **Sign up or log in** to [Weights & Biases (W&B)](https://wandb.ai).
2. Open the `default_wandb.yaml` file.
3. Replace the `entity` field with **your own W&B username or team name**.

This ensures your training runs are logged to your W&B account.

## Train the Model

To train the models, run this command:

For CERNN (Human): 
```train
python3 train.py 
```

For CERNN (Mouse):

If you want to train the model for the Mouse, you have to change "CERNN_mouse" model in train_CERNN_default.yaml, then run:
```train
python3 train.py 
```


This project uses [Hydra](https://hydra.cc/docs/intro/) for configuration management, allowing you to easily modify hyperparameters and other settings via the command line. All configuration files are located in the src/hydraconfigs directory.



## Results (Performance)

For Human -> ~95%

For Mouse -> ~ 91%

## Owner

- Hsu Yati Khin (oj24020@bristol.ac.uk)



# Cortically_Embedded_RNN
