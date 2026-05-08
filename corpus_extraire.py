"""
Ce script parcourt l'intégralité de Wikipédia en français via un streaming pour collecter 
des contextes d'utilisation de mots cibles.

-  Ce script utilise une stratégie de "saut" pour parcourir 
   le dataset du début à la fin, garantissant que les phrases proviennent de domaines variés.
-  Extraction limitée à 2 phrases par article pour éviter les biais thématiques.

Pour créer des fichiers .txt pour l'analyse de la polysémie via BERT.
"""


import nltk
from datasets import load_dataset
import re
import random
import os

# Initialisation des ressources linguistiques
nltk.download('wordnet')
nltk.download('omw-1.4')

def nettoyer_texte(texte):
    """Supprime les balises de référence et normalise les espaces."""
    texte = re.sub(r'\[\d+\]', '', texte)
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

def decouper_en_phrases(texte):
    """Découpe le texte en phrases avec des filtres de qualité."""
    # Découpage sur la ponctuation forte
    liste_brute = re.split(r'(?<=[.!?]) +|\n|;', texte)
    phrases_filtrees = []
    
    for s in liste_brute:
        s = s.strip()
        # Filtre de longueur (caractères)
        if 35 < len(s) < 500:
            mots = s.split()
            # Filtre de densité (nombre de mots)
            if 5 < len(mots) < 80:
                phrases_filtrees.append(s)
    return phrases_filtrees

def main():
    saisie_utilisateur = input("Entrez les mots à chercher (séparés par des virgules) : ")
    mots_a_extraire = [m.strip().lower() for m in saisie_utilisateur.split(",")]
    
    objectif_par_mot = 500  # Nombre de phrases cible
    LIMITE_PAR_ARTICLE = 2  # Diversité maximale par source
    PAS_DE_SAUT = 30        # Stratégie de saut pour couvrir tout Wikipédia

    # Initialisation des réservoirs de données
    corpus_final = {m: [] for m in mots_a_extraire}
    articles_lus = 0

    print(f"\n Connexion au flux Wikipédia (Français)...")
    print(f"Mode echantillonnage global (Saut : {PAS_DE_SAUT} articles)")
    
    try:
        # Chargement en mode streaming
        dataset = load_dataset("wikimedia/wikipedia", "20231101.fr", streaming=True, split="train")
        # Mélange local pour augmenter l'aléa
        dataset = dataset.shuffle(seed=42, buffer_size=1000)
    except Exception as erreur:
        print(f"Erreur lors du chargement : {erreur}")
        return

    # --- Boucle de parcours du corpus ---
    for entree in dataset:
        articles_lus += 1
        
        # Logique de saut pour la distribution globale
        if articles_lus % PAS_DE_SAUT != 0:
            continue
            
        contenu = entree['text']
        contenu_minuscule = contenu.lower()
        
        # Vérification des mots restants à compléter
        mots_actifs = [m for m in mots_a_extraire if len(corpus_final[m]) < objectif_par_mot]
        
        # Arrêt prématuré si tous les réservoirs sont pleins
        if not mots_actifs:
            break
            
        # Filtrage des mots présents dans cet article
        mots_trouves_ici = [m for m in mots_actifs if m in contenu_minuscule]
        
        if mots_trouves_ici:
            phrases_candidates = decouper_en_phrases(nettoyer_texte(contenu))
            
            for mot in mots_trouves_ici:
                # Recherche par expression régulière (mot entier uniquement)
                correspondances = [p for p in phrases_candidates if re.search(rf'\b{mot}\b', p.lower())]
                
                if correspondances:
                    random.shuffle(correspondances)
                    # Calcul de la quantité à prélever
                    reste_a_prendre = objectif_par_mot - len(corpus_final[mot])
                    quota = min(len(correspondances), LIMITE_PAR_ARTICLE, reste_a_prendre)
                    corpus_final[mot].extend(correspondances[:quota])
        
        # Affichage de la progression en temps réel
        progression = " | ".join([f"{m}: {len(corpus_final[m])}" for m in mots_a_extraire])
        print(f"  Article #{articles_lus} | {progression}", end="\r")

    print(f"\n\nÉchantillonnage terminé sur l'ensemble du corpus !")
    
    # --- Sauvegarde des résultats ---
    for mot, données in corpus_final.items():
        nom_fichier = f"corpus_{mot}.txt"
        with open(nom_fichier, "w", encoding="utf-8") as f:
            f.write("\n".join(données))
        print(f"'{mot}' : {len(données)} phrases sauvegardées ({nom_fichier})")

    os._exit(0)

if __name__ == "__main__":
    main()