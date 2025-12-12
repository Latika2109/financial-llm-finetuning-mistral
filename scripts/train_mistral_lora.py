import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from huggingface_hub import login

# Authenticate with Hugging Face Hub
login()  # Remove token from code for security; use HF CLI or env variable instead

# Load Pretrained Base Model
model_name = "mistralai/Mistral-7B-v0.1"

# Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Quantization Config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load Model with QLoRA
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto"
)

model.resize_token_embeddings(len(tokenizer))


# Load JSON Dataset
def load_json_dataset(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return Dataset.from_list(data)


dataset = load_json_dataset("data/fixed_dataset.json")


# Tokenization Function
def tokenize_function(examples):
    instructions = examples["instruction"]
    responses = examples["response"]

    # Ensure everything is string
    instruction_strs = [instr if isinstance(instr, str) else " ".join(instr) for instr in instructions]
    response_strs = [resp if isinstance(resp, str) else " ".join(resp) for resp in responses]

    texts = [instr + " " + resp for instr, resp in zip(instruction_strs, response_strs)]

    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512
    )


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["instruction", "response"])


# LoRA Configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


# Training Arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    weight_decay=0.01,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    logging_steps=100,
    save_steps=500,
    output_dir="outputs/mistral_lora_finance",
    report_to="none"
)


# Training
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args
)

trainer.train()


# Save LoRA Adapter
save_dir = "outputs/mistral_lora_finance_adapter"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Training complete. Adapter saved at:", save_dir)
