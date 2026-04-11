# Fine-tuning Llama 2 com LoRA/QLoRA

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

MODELO_BASE = "NousResearch/Llama-2-7b-hf"
R = 64
ALPHA = 16
DROPOUT = 0.1
EPOCHS = 3
BATCH = 4
LR = 2e-4
MAX_SEQ = 256
WARMUP = 0.03

cfg_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

modelo = AutoModelForCausalLM.from_pretrained(
    MODELO_BASE,
    quantization_config=cfg_bnb,
    device_map="auto"
)
tok = AutoTokenizer.from_pretrained(MODELO_BASE)
tok.pad_token = tok.eos_token
tok.padding_side = "right"

cfg_lora = LoraConfig(
    task_type="CAUSAL_LM",
    r=R,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    target_modules=["q_proj", "v_proj"]
)
modelo = get_peft_model(modelo, cfg_lora)

ds = load_dataset("json", data_files="musica.jsonl", split="train")


def formata(ex):
    return {"text": f"### Pergunta:\n{ex['prompt']}\n\n### Resposta:\n{ex['response']}"}


ds = ds.map(formata, remove_columns=ds.column_names)
n = int(len(ds) * 0.9)
ds_treino = ds.select(range(n))
ds_teste = ds.select(range(n, len(ds)))

args = SFTConfig(
    output_dir="adapter",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP,
    optim="paged_adamw_32bit",
    logging_steps=5,
    save_strategy="epoch",
    max_length=MAX_SEQ,
)


if __name__ == "__main__":
    modelo.print_trainable_parameters()

    trainer = SFTTrainer(
        model=modelo,
        processing_class=tok,
        train_dataset=ds_treino,
        eval_dataset=ds_teste,
        args=args,
    )

    print("treinando...")
    trainer.train()

    trainer.model.save_pretrained("adapter")
    print("adapter salvo em adapter/")
