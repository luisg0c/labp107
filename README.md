# Lab 7 - Fine-tuning com LoRA/QLoRA

Fine-tuning do Llama 2 7B com LoRA usando dataset sintetico de musica eletronica gerado via API da OpenAI.
O modelo eh carregado em 4-bit (QLoRA com bitsandbytes) e treinado com SFTTrainer da biblioteca trl.

## Dependencias

    pip install torch transformers peft trl bitsandbytes datasets openai

## Como rodar

Gerar o dataset (precisa de OPENAI_API_KEY no .env):

    python gera_dataset.py

Rodar o fine-tuning (precisa de GPU):

    python finetune.py

## Uso de IA

Ferramenta usada: Claude Sonnet 4.6

- Elaboracao do prompt de sistema pra geracao do dataset com a API da OpenAI
- Debug do erro de pad_token no tokenizer do Llama 2
