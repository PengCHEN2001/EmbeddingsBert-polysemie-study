import os
import json
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from transformers import CamembertTokenizer, CamembertModel

# Chargement du modèle CamemBERT et de son tokenizer
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertModel.from_pretrained("camembert-base")
model.eval()  # mode inférence : désactive le dropout
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def lire_corpus(fichier_txt):
    # Lit le fichier corpus_XXX.txt ligne par ligne
    # Chaque ligne est une phrase
    phrases = []
    with open(fichier_txt, encoding="utf-8") as f:
        for ligne in f:
            phrase = ligne.strip()
            if not ligne:
                continue
            if phrase:
                phrases.append(phrase)
    return phrases


def extraire_embedding(phrase, mot):
    # Étape 1 : tokeniser la phrase complète
    # Le tokenizer découpe la phrase en sous-mots (wordpieces)
    # Ex : "caporal" → ["cap", "oral"]
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Étape 2 : tokeniser le mot cible seul pour connaître sa représentation en tokens
    mot_tokens = tokenizer.encode(mot, add_special_tokens=False)
    input_ids = inputs["input_ids"][0].tolist()

    # Étape 3 : localiser la position du mot cible dans la séquence de tokens de la phrase
    # On cherche la sous-séquence mot_tokens dans input_ids
    mot_indices = []
    for i in range(len(input_ids) - len(mot_tokens) + 1):
        if input_ids[i : i + len(mot_tokens)] == mot_tokens:
            mot_indices = list(range(i, i + len(mot_tokens)))
            break

    # Si le mot n'est pas trouvé dans les tokens, on ignore la phrase
    if not mot_indices:
        return None

    # Étape 4 : passage dans le modèle
    # torch.no_grad() désactive le calcul du gradient (inutile en inférence)
    with torch.no_grad():
        outputs = model(**inputs)

    # Étape 5 : extraire le vecteur contextuel
    # last_hidden_state contient un vecteur de 768 dimensions par token
    # On prend les tokens du mot cible et on fait leur moyenne
    hidden = outputs.last_hidden_state[0]              # (nb_tokens, 768)
    return hidden[mot_indices].mean(dim=0).cpu().numpy()  # (768,)


def calculer_scores(embeddings):
    # Calcule toutes les similarités cosinus 2 à 2
    # Pour N embeddings : N*(N-1)/2 paires
    # Ex : 60 phrases → 1770 paires
    sims = [
        np.dot(embeddings[i], embeddings[j]) /
        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
        for i, j in combinations(range(len(embeddings)), 2)
    ]
    sims = np.array(sims)
    return {
        "sim_mean": round(float(sims.mean()), 4),
        "sim_std":  round(float(sims.std()),  4),  # score de polysémie
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
    phrases_ok = []

    for phrase in phrases:
        emb = extraire_embedding(phrase, mot)
        if emb is not None:
            embeddings.append(emb)
            phrases_ok.append(phrase)

    if len(embeddings) < 2:
        print(f"  Pas assez d'embeddings pour {mot}, ignoré.")
        continue

    # Sauvegarde des embeddings dans un fichier JSON
    # Format : liste de {"phrase": ..., "embedding": [768 floats]}
    json_data = [
        {"phrase": phrase, "embedding": emb.tolist()}
        for phrase, emb in zip(phrases_ok, embeddings)
    ]
    nom_json = f"embeddings_{mot}.json"
    with open(nom_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    scores = calculer_scores(embeddings)
    scores["mot"] = mot
    scores["nb_phrases"] = len(embeddings)
    resultats.append(scores)

    print(f"  mean={scores['sim_mean']}  std={scores['sim_std']}  "
          f"min={scores['sim_min']}  max={scores['sim_max']}")

# Sauvegarde de tous les scores dans un fichier CSV, trié par sim_std décroissant
df_out = pd.DataFrame(resultats)[["mot", "nb_phrases", "sim_mean", "sim_std", "sim_min", "sim_max"]]
df_out = df_out.sort_values("sim_std", ascending=False).reset_index(drop=True)
df_out.to_csv("scores.csv", index=False, encoding="utf-8")

print("\nscores.csv sauvegardé !")
print(df_out.to_string(index=False))