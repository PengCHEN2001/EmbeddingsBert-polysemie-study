import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn

# 1 : Comptage des sens via WordNet (synsets en français)

def compter_sens_wordnet(mot):
    synsets = wn.synsets(mot, lang="fra")
    return len(synsets)

# =============================================================
# Étape 2 : Application sur tous les mots du CSV
# =============================================================
df   = pd.read_csv("scores.csv")
mots = df["mot"].tolist()

print("=== Récupération des sens depuis WordNet (lang=fra) ===\n")
nb_sens_dict = {}

for mot in mots:
    nb = compter_sens_wordnet(mot)
    nb_sens_dict[mot] = nb if nb > 0 else None
    print(f"  [OK] '{mot}' → {nb} synsets")

df["nb_sens_wn"] = df["mot"].map(nb_sens_dict)
df.to_csv("scores_avec_wordnet.csv", index=False, encoding="utf-8")

print("\n=== Données ===")
print(df[["mot", "sim_std", "nb_sens_wn"]].to_string(index=False))


# 3 : Corrélation

df_clean  = df.dropna(subset=["nb_sens_wn"]).copy()

if len(df_clean) < 3:
    print("\nPas assez de données pour calculer une corrélation.")
    sys.exit()

std_vals  = df_clean["sim_std"].values
sens_vals = df_clean["nb_sens_wn"].values

r_p, p_p = pearsonr(std_vals, sens_vals)
r_s, p_s = spearmanr(std_vals, sens_vals)

print(f"\n{'='*45}")
print(f"  Corrélation de Pearson  : r = {r_p:.3f}  (p = {p_p:.3f})")
print(f"  Corrélation de Spearman : ρ = {r_s:.3f}  (p = {p_s:.3f})")
print(f"{'='*45}")


# 4 : Visualisation

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(sens_vals, std_vals, color="steelblue", edgecolors="k",
           linewidths=0.5, s=80, zorder=3)

for _, row in df_clean.iterrows():
    ax.annotate(
        row["mot"],
        xy=(row["nb_sens_wn"], row["sim_std"]),
        xytext=(5, 3),
        textcoords="offset points",
        fontsize=9
    )

m, b   = np.polyfit(sens_vals, std_vals, 1)
x_line = np.linspace(sens_vals.min() - 0.5, sens_vals.max() + 0.5, 100)
ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5,
        linestyle="--", label="régression linéaire")

ax.set_xlabel("Nombre de synsets (WordNet)", fontsize=11)
ax.set_ylabel("Score de polysémie BERT (sim_std)", fontsize=11)
ax.set_title(
    f"Corrélation sim_std ~ nb_synsets\nPearson r={r_p:.2f}  Spearman ρ={r_s:.2f}",
    fontsize=12
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("correlation_wordnet.png", dpi=150)
plt.show()
print("\ncorrelation_wordnet.png sauvegardé")
