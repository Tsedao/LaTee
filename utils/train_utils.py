import os

import numpy as np
import torch
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


def load_model(
        model_name, 
        cache_dir='~/.cache/huggingface/hub',
        device = 'cuda:0'
    ):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
   

    tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    device_map=device ,
                )
    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        # torch_dtype=torch.bfloat16,
                        quantization_config=bnb_config,
                        trust_remote_code = True,
                        device_map=device ,
                        cache_dir=cache_dir,
                    )
    
    return model, tokenizer

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def get_lora_model(model, **kwargs):

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=kwargs['r'],
        lora_alpha=kwargs['lora_alpha'],
        target_modules= ["k_proj", "v_proj"] ,
        lora_dropout=kwargs['lora_dropout'],
        bias="none",
        modules_to_save=["classifier"],
    )
    inference_model = get_peft_model(model, lora_config)

    print_trainable_parameters(inference_model)

    return inference_model

def calc_rank(label, pred):
    """

    Args:
        label: [N]
            Label index.
        pred: [N, n_cat]
            Predicted values.

    Returns:

    """
    label = np.array(label)
    pred = np.array(pred)
    if len(label.shape) < 2:
        label = label[:, None]
    if len(pred.shape) > 2:
        pred = pred.reshape(pred.shape[0], pred.shape[-1])

    label_val = np.take_along_axis(pred, label, axis=1)
    big_num = (pred > label_val).sum(axis=-1)
    equal_num = np.maximum((pred == label_val).sum(axis=-1), 1)

    rank = big_num + equal_num
    return rank