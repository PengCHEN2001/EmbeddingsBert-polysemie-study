import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

df = pd.read_csv("scores.csv")

# Nombre de sens(on ne peut pas scraper les informations de ce site automatiquement, on a compter les nombres de sens manuellement)

nb_sens = {
    "photosynthèse": 1,
    "appendicite": 1,
    "hélicoptère": 1,
    "chlorophylle": 1,
    "oxymètre": 1,
    "glace": 8,
    "carte": 12,
    "chef": 9,
    "bureau": 7,
    "grue": 3,
    "souris": 5,
    "plante": 6,
    "pièce": 7,
    "volant": 13,
    "langue": 6,
    "vague": 5,
    "pointe": 10,
    "mousse": 7
}

df["nb_sens_wikt"] = df["mot"].map(nb_sens)
df.to_csv("scores_avec_acad.csv", index=False, encoding="utf-8")
print(df[["mot", "sim_std", "nb_sens_wikt"]].to_string(index=False))


# Corrélation

df_clean = df.dropna(subset=["nb_sens_wikt"])
std_vals  = df_clean["sim_std"].values
sens_vals = df_clean["nb_sens_wikt"].values

r_p, p_p = pearsonr(std_vals, sens_vals)
r_s, p_s = spearmanr(std_vals, sens_vals)

print(f"\n{'='*45}")
print(f"  Corrélation de Pearson  : r = {r_p:.3f}  (p = {p_p:.3f})")
print(f"  Corrélation de Spearman : ρ = {r_s:.3f}  (p = {p_s:.3f})")
print(f"{'='*45}")


# Visualisation

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(sens_vals, std_vals, color="steelblue", edgecolors="k", linewidths=0.5, s=80, zorder=3)

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
ax.plot(x_line, m * x_line + b, color="tomato", linewidth=1.5, linestyle="--", label="régression linéaire")

ax.set_xlabel("Nombre de sens (dictionnnaire-academie)", fontsize=11)
ax.set_ylabel("Score de polysémie BERT (sim_std)", fontsize=11)
ax.set_title(f"Corrélation sim_std ~ nb_sens\nPearson r={r_p:.2f}  Spearman ρ={r_s:.2f}", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("correlation_academie.png", dpi=150)
plt.show()
print("\ncorrelation_academie.png sauvegardé")

