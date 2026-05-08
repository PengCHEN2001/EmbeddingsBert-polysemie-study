#!/usr/bin/env python3
import sys
import time
import requests
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


# Configuration BabelNet

API_KEY  = "49242691-98e7-4349-8001-144391f5942f"
API_URL  = "https://babelnet.io/v9/getSynsetIds"
HEADERS  = {"User-Agent": "PolysemieProject/1.0"}


# Étape 1 : Comptage des sens via BabelNet
# On récupère tous les synsets pour un mot en français,
# puis on filtre sur la source WN (WordNet) ou WIKI selon besoin

def compter_sens_babelnet(mot):
    params = {
        "lemma":    mot,
        "searchLang": "FR",       # langue de recherche : français
        "targetLang": "FR",       # langue des résultats
        "key":      API_KEY,
    }
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=15)

        # Gestion du quota (limite : 1000 requêtes/jour en version gratuite)
        if resp.status_code == 403:
            print(f"  [QUOTA] Limite API atteinte pour '{mot}'")
            return None

        resp.raise_for_status()
        data = resp.json()

        # data est une liste de synsets : [{"id": "...", "pos": "NOUN", ...}, ...]
        if not data:
            return 0

        # On peut filtrer par catégorie grammaticale si besoin
        # Ici on garde tous les synsets toutes POS confondues
        nb = len(data)
        return nb

    except Exception as e:
        print(f"  [ERREUR] '{mot}' : {e}")
        return None


# Étape 2 : Application sur tous les mots du CSV

df   = pd.read_csv("scores.csv")
mots = df["mot"].tolist()

print("=== Récupération des sens depuis BabelNet (FR) ===\n")
nb_sens_dict = {}

for mot in mots:
    nb = compter_sens_babelnet(mot)
    nb_sens_dict[mot] = nb if nb and nb > 0 else None
    print(f"  [OK] '{mot}' → {nb} synsets")
    time.sleep(1)

df["nb_sens_bn"] = df["mot"].map(nb_sens_dict)
df.to_csv("scores_avec_babelnet.csv", index=False, encoding="utf-8")

print("\n=== Données ===")
print(df[["mot", "sim_std", "nb_sens_bn"]].to_string(index=False))


# Étape 3 : Corrélation

df_clean  = df.dropna(subset=["nb_sens_bn"]).copy()

if len(df_clean) < 3:
    print("\nPas assez de données pour calculer une corrélation.")
    sys.exit()

std_vals  = df_clean["sim_std"].values
sens_vals = df_clean["nb_sens_bn"].values

r_p, p_p = pearsonr(std_vals, sens_vals)
r_s, p_s = spearmanr(std_vals, sens_vals)

print(f"\n{'='*45}")
print(f"  Corrélation de Pearson  : r = {r_p:.3f}  (p = {p_p:.3f})")
print(f"  Corrélation de Spearman : ρ = {r_s:.3f}  (p = {p_s:.3f})")
print(f"{'='*45}")

# Étape 4 : Visualisation

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(sens_vals, std_vals, color="steelblue", edgecolors="k",
           linewidths=0.5, s=80, zorder=3)

for _, row in df_clean.iterrows():
    ax.annotate(
        row["mot"],
        xy=(row["nb_sens_bn"], row["sim_std"]),
        xytext=(5, 3),
        textcoords="offset points",
        fontsize=9
    )

m, b   = np.polyfit(sens_vals, std_vals, 1)
x_line = np.linspace(sens_vals.min() - 0.5, sens_vals.max() + 0.5, 100)
ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5,
        linestyle="--", label="régression linéaire")

ax.set_xlabel("Nombre de synsets (BabelNet)", fontsize=11)
ax.set_ylabel("Score de polysémie BERT (sim_std)", fontsize=11)
ax.set_title(
    f"Corrélation sim_std ~ nb_synsets BabelNet\nPearson r={r_p:.2f}  Spearman ρ={r_s:.2f}",
    fontsize=12
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("correlation_babelnet.png", dpi=150)
plt.show()
print("\ncorrelation_babelnet.png sauvegardé")
