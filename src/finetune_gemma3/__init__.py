"""QLoRA finetuning of gemma3.

## Important Information:
The following notebook and its companion paper give explanation on how this is
to be done.
* https://huggingface.co/blog/4bit-transformers-bitsandbytes
* https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
"""
from datasets import (
    Dataset,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

modelname = "google/gemma-3-1b-it"

##### QUANTIZATION & MODEL LOADING ############################################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    modelname,
)
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    quantization_config = bnb_config,
    device_map="auto",
    attn_implementation="eager",
)

##### PEFT FINETUNING #########################################################
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
lora  = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)


##### DATASET #################################################################
def tokenize(sample):  # noqa: ANN001, ANN201
    """Turn sample into a prompt."""
    messages = [
        {"role": "system", "content": "Respond to this user email."},
        {"role": "user",   "content": sample["request"]},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors=None)

dataset = Dataset.from_csv("./resources/mails_dataset.csv")
dataset = dataset.select(range(10))
dataset = dataset.map(tokenize, batched=False)

##### ACTUAL FINETUNING #######################################################
args  = TrainingArguments(
    num_train_epochs            = 1,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 1,
    gradient_checkpointing      = True,
    overwrite_output_dir        = True,
    dataloader_num_workers= 1,
)
trainer = Trainer(
    model,
    args,
    train_dataset=dataset,
    #eval_dataset =None, # fixme
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
