import openai
import json
import os
import random

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Voce eh um especialista em musica eletronica. Gere um par de pergunta e resposta sobre musica eletronica.
A pergunta deve ser sobre subgeneros, producao musical, DJs, historia, equipamentos ou cultura.
A resposta deve ter 2-3 frases, em tom conversacional e informativo.
Retorne SOMENTE um JSON valido no formato: {"prompt": "...", "response": "..."}"""

N_EXEMPLOS = 60


def gera_par():
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Gere uma pergunta e resposta sobre musica eletronica."}
        ],
        temperature=0.9,
        max_tokens=300
    )
    txt = resp.choices[0].message.content.strip()
    return json.loads(txt)


if __name__ == "__main__":
    dados = []
    for i in range(N_EXEMPLOS):
        try:
            par = gera_par()
            dados.append(par)
            print(f"{i+1}/{N_EXEMPLOS}: {par['prompt'][:60]}...")
        except Exception as e:
            print(f"{i+1}/{N_EXEMPLOS}: erro - {e}")

    random.shuffle(dados)

    with open("musica.jsonl", "w", encoding="utf-8") as f:
        for d in dados:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n{len(dados)} pares salvos em musica.jsonl")
