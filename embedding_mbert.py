import os
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from transformers import BertTokenizer, BertModel

print("Chargement de mBERT...")
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
model = BertModel.from_pretrained("bert-base-multilingual-cased")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modèle chargé sur : {device}\n")


def lire_corpus(fichier_txt):
    phrases = []
    with open(fichier_txt, encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if not ligne:
                continue
            phrase = ligne.split("\t|")[0].strip()
            if phrase:
                phrases.append(phrase)
    return phrases


def extraire_embedding(phrase, mot):
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    mot_tokens = tokenizer.encode(mot, add_special_tokens=False)
    input_ids = inputs["input_ids"][0].tolist()

    mot_indices = []
    for i in range(len(input_ids) - len(mot_tokens) + 1):
        if input_ids[i : i + len(mot_tokens)] == mot_tokens:
            mot_indices = list(range(i, i + len(mot_tokens)))
            break

    if not mot_indices:
        return None

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state[0]
    return hidden[mot_indices].mean(dim=0).cpu().numpy()


def calculer_scores(embeddings):
    sims = [
        np.dot(embeddings[i], embeddings[j]) /
        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        for i, j in combinations(range(len(embeddings)), 2)
    ]
    sims = np.array(sims)
    return {
        "sim_mean": round(float(sims.mean()), 4),
        "sim_std":  round(float(sims.std()),  4),
        "sim_min":  round(float(sims.min()),  4),
        "sim_max":  round(float(sims.max()),  4),
    }


fichiers = sorted([f for f in os.listdir(".") if f.startswith("corpus_") and f.endswith(".txt")])

if not fichiers:
    print("Aucun fichier corpus_XXX.txt trouvé.")
    exit()

resultats = []

for fichier in fichiers:
    mot = fichier.replace("corpus_", "").replace(".txt", "")
    print(f"Traitement : {mot}")

    phrases = lire_corpus(fichier)
    embeddings = []

    for phrase in phrases:
        emb = extraire_embedding(phrase, mot)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) < 2:
        print(f"  Pas assez d'embeddings pour {mot}, ignoré.")
        continue

    scores = calculer_scores(embeddings)
    scores["mot"] = mot
    scores["nb_phrases"] = len(embeddings)
    resultats.append(scores)

    print(f"  mean={scores['sim_mean']}  std={scores['sim_std']}  "
          f"min={scores['sim_min']}  max={scores['sim_max']}")

df_out = pd.DataFrame(resultats)[["mot", "nb_phrases", "sim_mean", "sim_std", "sim_min", "sim_max"]]
df_out = df_out.sort_values("sim_std", ascending=False).reset_index(drop=True)
df_out.to_csv("scores_mbert.csv", index=False, encoding="utf-8")

print("\nscores_mbert.csv sauvegardé !")
print(df_out.to_string(index=False))
