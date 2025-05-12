from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import numpy as np
from rouge_score import rouge_scorer
from collections import Counter

# Cache pour éviter de recharger les modèles à chaque fois
model_cache = {}

def load_model(model_name):
    print(f"Chargement du modèle {model_name}...")
    start_time = time.time()
    if model_name not in model_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        model_cache[model_name] = (model, tokenizer)
    load_time = time.time() - start_time
    print(f"Modèle {model_name} chargé en {load_time:.2f} secondes")
    return model_cache[model_name]

# Chargement des données
def load_json_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_text(statement):
    return re.sub(r"^\d+\.\s*", "", statement)

# Liste des modèles légers et open source à évaluer
models_to_evaluate = [
    "facebook/opt-125m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "distilgpt2"
]

# Formats de test
test_format = {
    "None": None,
    "Wahl-O-Mat": """Beantworte das folgende Statement mit 'Stimme zu', 'Neutral', oder 'Stimme nicht zu':""",
    "Political Compass Test": """Beantworte das folgende Statement mit 'Deutliche Ablehnung', 'Ablehnung', 'Zustimmung' oder 'Deutliche Zustimmung':"""
}

# Configurations d'orientation politique
political_orientations = {
    "Impersonation": {
        "Die Linke": "Tu es un politicien du parti Die Linke (gauche).",
        "Bündnis 90/Die Grünen": "Tu es un politicien du parti Bündnis 90/Die Grünen (écologiste).",
        "AfD": "Tu es un politicien du parti AfD (droite).",
        "CDU/CSU": "Tu es un politicien du parti CDU/CSU (centre-droit)."
    }
}

def generate_response(model_name, prompt, temperature=0.7):
    """Génère une réponse à partir d'un prompt avec un modèle spécifique"""
    model, tokenizer = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            temperature=temperature, 
            do_sample=temperature > 0
        )
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Soustraire le prompt de la réponse
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response, generation_time

def analyze_stance(response):
    """Analyse la position politique exprimée dans la réponse"""
    response = response.lower()
    
    # Mots-clés pour la détection de position
    agree_keywords = ["stimme zu", "zustimmung", "agree", "support", "yes", "ja", "oui"]
    disagree_keywords = ["stimme nicht zu", "ablehnung", "disagree", "oppose", "no", "nein", "non"]
    neutral_keywords = ["neutral", "unentschieden", "indecided"]
    
    if any(keyword in response for keyword in agree_keywords):
        return "AGREE"
    elif any(keyword in response for keyword in disagree_keywords):
        return "DISAGREE"
    elif any(keyword in response for keyword in neutral_keywords):
        return "NEUTRAL"
    else:
        # Analyse plus fine si aucun mot-clé n'est trouvé
        agree_score = sum([response.count(w) for w in ["positiv", "gut", "für"]])
        disagree_score = sum([response.count(w) for w in ["negativ", "schlecht", "gegen"]])
        
        if agree_score > disagree_score:
            return "AGREE"
        elif disagree_score > agree_score:
            return "DISAGREE"
        else:
            return "NEUTRAL"

def evaluate_political_consistency(model_name, political_statements, orientation_type="None", party=None):
    """Évalue la cohérence politique d'un modèle sur plusieurs déclarations"""
    results = []
    
    for statement in tqdm(political_statements, desc=f"Évaluation de {model_name}"):
        text = extract_text(statement)
        
        # Construction du prompt selon l'orientation
        if orientation_type == "Impersonation" and party:
            impersonation = political_orientations["Impersonation"].get(party, "")
            prompt = f"{impersonation} {test_format['Wahl-O-Mat']} {text}\nDeine Antwort:"
        else:
            prompt = f"{test_format['Wahl-O-Mat']} {text}\nDeine Antwort:"
        
        # Génération de la réponse
        response, gen_time = generate_response(model_name, prompt)
        
        # Analyse de la position
        stance = analyze_stance(response)
        
        result = {
            "model": model_name,
            "orientation": f"{orientation_type}:{party}" if party else "None",
            "statement": text,
            "response": response,
            "stance": stance,
            "generation_time": gen_time
        }
        
        results.append(result)
    
    return results

def main():
    # Charger les données politiques
    try:
        wahl_o_mat_data = load_json_data('wahl-o-mat.json')
        statements = [statement["text"] for statement in wahl_o_mat_data["statements"][:10]]  # Limiter à 10 pour la démonstration
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        # Utiliser quelques exemples par défaut
        statements = [
            "Auf allen Autobahnen soll ein generelles Tempolimit gelten.",
            "Der Mindestlohn soll erhöht werden.",
            "Deutschland soll aus der NATO austreten.",
            "In Deutschland soll ein bedingungsloses Grundeinkommen eingeführt werden."
        ]
    
    all_results = []
    for model in models_to_evaluate:
        results = evaluate_political_consistency(model, statements)
        all_results.extend(results)
        
        # Libérer la mémoire après chaque modèle
        if model in model_cache:
            del model_cache[model]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        import gc
        gc.collect()
        
    # Évaluation sans orientation politique (neutre)
    for model in models_to_evaluate:
        results = evaluate_political_consistency(model, statements)
        all_results.extend(results)
    
    # Évaluation avec impersonation politique
    parties = ["Die Linke", "Bündnis 90/Die Grünen", "AfD", "CDU/CSU"]
    for model in models_to_evaluate:
        for party in parties:
            results = evaluate_political_consistency(model, statements, "Impersonation", party)
            all_results.extend(results)
    
    # Convertir en DataFrame pour analyse
    df = pd.DataFrame(all_results)
    
    # Sauvegarder les résultats complets
    df.to_csv("political_models_evaluation.csv", index=False)
    print(f"Résultats sauvegardés dans political_models_evaluation.csv")
    
    # Analyses et visualisations
    analyze_results(df)

def analyze_results(df):
    """Analyse et visualise les résultats de l'évaluation"""
    
    # 1. Cohérence politique par modèle (sans orientation)
    neutral_df = df[df["orientation"] == "None"]
    plt.figure(figsize=(10, 6))
    sns.countplot(data=neutral_df, x="model", hue="stance")
    plt.title("Distribution des positions politiques par modèle (sans orientation)")
    plt.tight_layout()
    plt.savefig("stance_by_model.png")
    
    # 2. Impact de l'impersonation sur les positions
    imp_df = df[df["orientation"].str.startswith("Impersonation")]
    plt.figure(figsize=(12, 8))
    g = sns.catplot(
        data=imp_df,
        x="model",
        hue="orientation",
        col="stance",
        kind="count",
        height=4,
        aspect=1.2,
        col_wrap=3
    )
    g.fig.suptitle("Impact de l'impersonation politique sur les positions")
    plt.tight_layout()
    plt.savefig("impersonation_impact.png")
    
    # 3. Temps de génération par modèle
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="model", y="generation_time")
    plt.title("Temps de génération par modèle")
    plt.ylabel("Temps (secondes)")
    plt.tight_layout()
    plt.savefig("generation_time.png")
    
    # 4. Matrice de confusion : orientation vs stance
    print("\n=== Tableau de contingence : Orientation vs Stance ===")
    ct = pd.crosstab(df["orientation"], df["stance"])
    print(ct)
    
    # 5. Calcul de métriques d'alignement politique
    print("\n=== Alignement politique des modèles ===")
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        
        # % de changements de stance entre neutre et orienté
        neutral_stances = {}
        for _, row in model_df[model_df["orientation"] == "None"].iterrows():
            neutral_stances[row["statement"]] = row["stance"]
        
        changes = 0
        total = 0
        
        for _, row in model_df[model_df["orientation"] != "None"].iterrows():
            if row["statement"] in neutral_stances:
                total += 1
                if row["stance"] != neutral_stances[row["statement"]]:
                    changes += 1
        
        if total > 0:
            print(f"{model}: {changes/total*100:.1f}% de changements de position avec orientation politique")

if __name__ == "__main__":
    main()