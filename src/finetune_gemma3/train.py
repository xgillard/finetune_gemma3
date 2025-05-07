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

output_dir    = "Llama-3.2-1B-Instruct-arkey_emails-qlora"
modelname     = "meta-llama/Llama-3.2-1B-Instruct"
epochs        = 5
dset_path     = "./anonymized.csv"
batch_size    = 32
accumulation  = 10     # virtual batches of 320 samples
checkpointing = False  # true si besoin de plus de ram.

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
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    modelname,
    quantization_config = bnb_config,
    device_map="auto",
    attn_implementation="eager",
)

##### PEFT FINETUNING #########################################################
if checkpointing:
    model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=checkpointing)
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
        {"role": "system", "content": "Respond to this user email using the given pieces of information only."},
        {"role": "user",   "content": f"{sample['request']}\n\n\n# Information you can use:\n{sample['context']}"},
        {"role": "assistant", "content": sample["response"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors=None)

dataset = Dataset.from_csv(dset_path)
dataset = dataset.map(tokenize, batched=False)

shards  = dataset.shuffle(seed=42).train_test_split(test_size=0.1)
trainds = shards["train"]
evalds  = shards["test"]

##### ACTUAL FINETUNING #######################################################
args  = TrainingArguments(
    output_dir                  = output_dir,
    num_train_epochs            = epochs,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    gradient_accumulation_steps = accumulation,
    gradient_checkpointing      = checkpointing,
    save_strategy               = "epoch",
    eval_strategy               = "epoch",
    overwrite_output_dir        = True,
    load_best_model_at_end      = True,
    push_to_hub                 = f"xaviergillard/{output_dir}",
)
trainer = Trainer(
    model,
    args,
    train_dataset=trainds,
    eval_dataset=evalds,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(f"final/{output_dir}")
model.push_to_hub(f"xaviergillard/{output_dir}")
