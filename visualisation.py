import json
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# Chargement des embeddings
# Format : { "mot": { "embeddings": [...], "sentences": [...] } }

with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

os.makedirs("visualisations", exist_ok=True)


def plot_word(word, entry):
    # Compatibilité avec l'ancien format liste simple
    if isinstance(entry, list):
        embeddings = entry
        sentences = [f"occurrence {i}" for i in range(len(entry))]
    else:
        embeddings = entry["embeddings"]
        sentences  = entry.get("sentences", [f"occurrence {i}" for i in range(len(embeddings))])

    X = np.array(embeddings)

    if len(X) < 3:
        print(f"  [IGNORÉ] {word} : occurrences insuffisantes ({len(X)})")
        return

    # Texte affiché au survol : retour à la ligne tous les 60 caractères pour lisibilité
    hover_texts = [
        "<br>".join([s[i:i+60] for i in range(0, len(s), 60)])
        for s in sentences
    ]

    n_plots = 3 if UMAP_AVAILABLE else 2
    subplot_titles = ["ACP (linéaire)", f"t-SNE", "UMAP"] [:n_plots]
    fig = make_subplots(rows=1, cols=n_plots, subplot_titles=subplot_titles)


    # 1. ACP — méthode linéaire
    #    Affiche aussi le pourcentage de variance expliquée sur chaque axe

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100

    fig.add_trace(go.Scatter(
        x=X_pca[:, 0], y=X_pca[:, 1],
        mode="markers",
        marker=dict(size=4, opacity=0.7, line=dict(width=0.3, color="black")),
        text=hover_texts,
        hovertemplate="<b>ACP</b><br>%{text}<extra></extra>",
        name="ACP"
    ), row=1, col=1)

    fig.update_xaxes(title_text=f"CP1 ({var[0]:.1f}% var.)", row=1, col=1)
    fig.update_yaxes(title_text=f"CP2 ({var[1]:.1f}% var.)", row=1, col=1)


    # 2. t-SNE — méthode non-linéaire, voisinages locaux

    perplexity = min(50, len(X) // 10)
    X_tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca", # plus stable que init="random"
        random_state=42,
        learning_rate="auto"
    ).fit_transform(X)

    fig.add_trace(go.Scatter(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        mode="markers",
        marker=dict(size=4, opacity=0.7, line=dict(width=0.3, color="black")),
        text=hover_texts,
        hovertemplate="<b>t-SNE</b><br>%{text}<extra></extra>",
        name=f"t-SNE (perp.={perplexity})"
    ), row=1, col=2)


    # 3. UMAP — non-linéaire, préserve structure locale ET globale

    if UMAP_AVAILABLE:
        n_neighbors = min(30, len(X) - 1)
        X_umap = umap.UMAP(n_neighbors=n_neighbors, random_state=42).fit_transform(X)

        fig.add_trace(go.Scatter(
            x=X_umap[:, 0], y=X_umap[:, 1],
            mode="markers",
            marker=dict(size=4, opacity=0.5, line=dict(width=0.3, color="black")),
            text=hover_texts,
            hovertemplate="<b>UMAP</b><br>%{text}<extra></extra>",
            name=f"UMAP (n_neighbors={n_neighbors})"
        ), row=1, col=3)

    fig.update_layout(
        title_text=f"« {word} » — {len(X)} occurrences",
        title_font_size=14,
        height=600,
        width=650 * n_plots,
        showlegend=False
    )

    # Sauvegarde en HTML interactif (hover fonctionnel)
    out_path = f"visualisations/{word}_visualisation.html"
    fig.write_html(out_path)
    print(f"  [OK] {word} → {out_path}")



# Boucle principale

for word, entry in data.items():
    print(f"Visualisation : {word}")
    plot_word(word, entry)

print("\nTerminé → graphiques sauvegardés dans visualisations/")
