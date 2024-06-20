# Latent Logic Tree Extraction for Events Explaination (LaTee)

The code was tested on Ubuntu 22.04, python-3.10.14, CUDA 12.3
## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/Tsedao/LaTee.git
    cd LaTee
    ```

2. **Create a virtual environment (optional but recommended)**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `venv\Scripts\activate`
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

### Training

First config your huggingface cache dir `cache_dir` and wandb-api key in `train_lltot.py`,
Start trainning by calling
```bash
python train_lltot.py \
    --gpu=1 \
    --dataset=synthetic5 \   # or use your customized dataset under folder data/
    --max_depth=3 \   
    --topk=3 \               # expand nodes per level (topk <= max_width)
    --max_width=5 \          # down sampling event numbers 
    --bs=8  \ 
    --logic_model_lr=0.001 \ # logic model learning rate  
    --alternate_every=1  \   # Update frequencies between E step and M step
    --llm_size=medium \      # E-step LLM size (tunable) opt-1.3b as medium, opt-6.7b as large 
    --epoch=10 \
    --llm_lr=1e-5 \ 
    --lm_update_steps=1 \    # gradient update steps per M step    
    --inf_llm_size=zephyr-3b \  # M-step LLM size (forzen)
    --explore \ 
    --warmup  \              # warmup LM learning rate from zero
    --learn_prior \       
    --seed=112 
``` 
