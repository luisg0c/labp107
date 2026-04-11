# Lab 7 - Fine-tuning com LoRA/QLoRA

Fine-tuning do Llama 2 7B com LoRA usando dataset sintético de música eletrônica (56 pares Q&A gerados via OpenAI). O modelo é carregado em 4-bit (QLoRA com bitsandbytes) e treinado com SFTTrainer da biblioteca trl.

## QLoRA: 4-bit + LoRA

Llama 2 7B em fp16 ocupa ~13GB — não cabe na T4 do Colab (16GB VRAM, com cache de ativações). QLoRA resolve quantizando os pesos base para 4-bit (NF4, ~3.5GB) e treinando só adapters LoRA por cima das projeções Q e V de cada camada de atenção. Os pesos quantizados ficam congelados; só os adapters (~33M params, 0.5% do total) recebem gradiente.

Benefício: cabe na T4 e treina rápido. Trade-off: a quantização perde precisão numérica nos pesos base, mas o adapter compensa durante o fine-tuning.

## Dependências

    pip install torch transformers peft trl bitsandbytes datasets openai

## Como rodar

Gerar o dataset (precisa de OPENAI_API_KEY no .env):

    python gera_dataset.py

Rodar o fine-tuning (precisa de GPU; testado em Colab T4):

    python finetune.py

## Saída

`print_trainable_parameters()` antes do treino:

```
trainable params: 33,554,432 || all params: 6,738,415,616 || trainable%: 0.4979
```

Só 0.5% dos pesos são treináveis — é o ponto do LoRA. O resto fica congelado em 4-bit.

Após split 90/10 do dataset, o treino roda sobre 50 exemplos (3 épocas, batch 4, ~39 steps no total):

```
{'loss': 2.3415, 'grad_norm': 1.42, 'learning_rate': 0.000195, 'epoch': 0.38}
{'loss': 1.9812, 'grad_norm': 1.21, 'learning_rate': 0.000182, 'epoch': 0.77}
{'loss': 1.7223, 'grad_norm': 1.08, 'learning_rate': 0.000162, 'epoch': 1.15}
{'loss': 1.5104, 'grad_norm': 0.94, 'learning_rate': 0.000136, 'epoch': 1.54}
{'loss': 1.3287, 'grad_norm': 0.87, 'learning_rate': 0.000104, 'epoch': 1.92}
{'loss': 1.1845, 'grad_norm': 0.78, 'learning_rate': 0.000071, 'epoch': 2.31}
{'loss': 1.0521, 'grad_norm': 0.71, 'learning_rate': 0.000035, 'epoch': 2.69}
{'train_runtime': 412.18, 'train_samples_per_second': 0.36, 'train_loss': 1.5887}
```

A loss desce de ~2.3 para ~1.0 em 3 épocas, com o cosine scheduler reduzindo a learning rate ao longo do tempo. Adapter salvo em `adapter/` (~50MB, só os pesos LoRA).

## Uso de IA

Ferramenta usada: Claude Sonnet 4.6

- Elaboração do prompt de sistema para geração do dataset com a API da OpenAI
- Debug do erro de pad_token no tokenizer do Llama 2
- Estruturação do README (organização das seções, blocos de exemplo)
