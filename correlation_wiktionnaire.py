import re
import sys
import time
import requests
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


# Configuration

API     = "https://fr.wiktionary.org/w/api.php"
HEADERS = {"User-Agent": "SensCounter/1.0 (educational script)"}


# 1 : Récupération du wikitext depuis le Wiktionnaire

def fetch_wikitext(mot):
    resp = requests.get(
        API,
        params={"action": "parse", "page": mot, "prop": "wikitext", "format": "json"},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        return None
    return data["parse"]["wikitext"]["*"]


# 2 : Comptage des sens dans la section française

def compter_sens_francais(wikitext):
    fr_match = re.search(
        r"==\s*\{\{(?:langue\|fr|=fr=)\}\}\s*==(.+?)(?:==\s*\{\{(?:langue\||=)[a-z]|$)",
        wikitext, re.DOTALL,
    )
    section_fr = fr_match.group(1) if fr_match else wikitext

    pattern_section = re.compile(
        r"===\s*\{\{S\|([^|}\n]+)(?:\|[^}]*)?\}\}\s*===", re.IGNORECASE
    )
    sections = pattern_section.split(section_fr)

    total = 0
    for i in range(1, len(sections), 2):
        bloc = sections[i + 1] if i + 1 < len(sections) else ""
        for ligne in bloc.splitlines():
            if re.match(r"^#\s+", ligne) and not re.match(r"^##", ligne):
                total += 1
    return total


# 3 : Application sur tous les mots du CSV

df = pd.read_csv("scores.csv")
mots = df["mot"].tolist()

print("=== Récupération des sens depuis le Wiktionnaire ===\n")
nb_sens_dict = {}

for mot in mots:
    try:
        wikitext = fetch_wikitext(mot)
        if wikitext is None:
            print(f"  [ABSENT] '{mot}'")
            nb_sens_dict[mot] = None
        else:
            nb = compter_sens_francais(wikitext)
            nb_sens_dict[mot] = nb if nb > 0 else None
            print(f"  [OK] '{mot}' → {nb} sens")
    except Exception as e:
        print(f"  [ERREUR] '{mot}' : {e}")
        nb_sens_dict[mot] = None
    time.sleep(5)  # politesse envers l'API

df["nb_sens_wikt"] = df["mot"].map(nb_sens_dict)
df.to_csv("scores_avec_wikt.csv", index=False, encoding="utf-8")

print("\n=== Données ===")
print(df[["mot", "sim_std", "nb_sens_wikt"]].to_string(index=False))

# 4 : Corrélation

df_clean = df.dropna(subset=["nb_sens_wikt"])

if len(df_clean) < 3:
    print("\nPas assez de données pour calculer une corrélation.")
    sys.exit()

std_vals  = df_clean["sim_std"].values
sens_vals = df_clean["nb_sens_wikt"].values

r_p, p_p = pearsonr(std_vals, sens_vals)
r_s, p_s = spearmanr(std_vals, sens_vals)

print(f"\n{'='*45}")
print(f"  Corrélation de Pearson  : r = {r_p:.3f}  (p = {p_p:.3f})")
print(f"  Corrélation de Spearman : ρ = {r_s:.3f}  (p = {p_s:.3f})")
print(f"{'='*45}")


# 5 : Visualisation

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(sens_vals, std_vals, color="steelblue", edgecolors="k",
           linewidths=0.5, s=80, zorder=3)

for _, row in df_clean.iterrows():
    ax.annotate(
        row["mot"],
        xy=(row["nb_sens_wikt"], row["sim_std"]),
        xytext=(5, 3),
        textcoords="offset points",
        fontsize=9
    )

m, b = np.polyfit(sens_vals, std_vals, 1)
x_line = np.linspace(sens_vals.min() - 0.5, sens_vals.max() + 0.5, 100)
ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5,
        linestyle="--", label="régression linéaire")

ax.set_xlabel("Nombre de sens (Wiktionnaire)", fontsize=11)
ax.set_ylabel("Score de polysémie BERT (sim_std)", fontsize=11)
ax.set_title(
    f"Corrélation sim_std ~ nb_sens\nPearson r={r_p:.2f}  Spearman ρ={r_s:.2f}",
    fontsize=12
)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("correlation_wiktionnaire.png", dpi=150)
plt.show()
print("\ncorrelation_wiktionnaire.png sauvegardé")
