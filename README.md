# Latent Logic Tree Extraction for Events Explaination (LaTee)

The code is tested on Ubuntu 22.04, python-3.10.14, CUDA 12.3
## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/yourproject.git
    cd yourproject
    ```

2. **Create a virtual environment (optional but recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Dataset preparation
Prepare your dataset under `data` folder. For each dataset, it should contains three files:
1. a `#YOUR_DATASET#_configs.json` file that configs target events and body events, it should at least contains:,
```py
    {
        "head_preds": list[int], # a list of target events ids as head predicates 
        "body_preds": list[int], # a list of body events ids as body predicates, we do not predict them 
        "head_preds_name": list[str], # a list of target events names corresponding to head_preds 
        "body_preds_name": list[str]. # a list of body events names corresponding to body_preds
    }
```
2. a `#YOUR_DATASET#_x.pkl` file that contains historical event sequences of K trajectories, with format of
```py
    x = [
        ## This is the first trajectory
        [
            [Event Time 0, Event ID 0],
            [Event Time 1, Event ID 1]
            ...
        ]
        ## This is the second trajectory
        [
            [Event Time 0, Event ID 0],
            [Event Time 1, Event ID 1]
            ...
        ]
        ...
    ]
```
3. a `#YOUR_DATASET#_y.pkl` file that contains the final event type as ID of the k trajectories in `#YOUR_DATASET#_x.pkl` with format of
```py
   y = list[int]          # length of y should equal to x
```

### Trainning

First config your huggingface cache dir `cache_dir` and wandb key in `train_lltot.py`,
Start trainning by calling
```bash
python train_lltot.py \
    --gpu=1 \
    --dataset=synthetic5 \ # or use your customized dataset under folder data/
    --max_depth=3 \
    --topk=3 \
    --max_width=5 \ 
    --bs=8  \
    --logic_model_lr=0.001 \ 
    --alternate_every=1  \
    --llm_size=medium \
    --epoch=10 \
    --llm_lr=1e-5 \ 
    --lm_update_steps=1 \ 
    --inf_llm_size=zephyr-3b \ 
    --explore \ 
    --warmup  \
    --seed=112 \
    --learn_priored=112 \ 
    --learn_prior
``` 
