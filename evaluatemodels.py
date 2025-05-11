from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

models_to_evaluate = [
    {
        "name": "TinyLlama-1.1B",
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_adapter": "TinyLlama/tinyllama-chat-1.1b-v1.0-lora"
    },
    {
        "name": "TinyLlama-1.1B (sans LoRA)",
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_adapter": None
    },
    {
        "name": "Phi-2",
        "base_model": "microsoft/phi-2",
        "lora_adapter": None
    },
    {
        "name": "Mistral-7B (version lite)",
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2", 
        "lora_adapter": None
    },
    {
        "name": "BLOOM-1.7B",
        "base_model": "bigscience/bloom-1b7",
        "lora_adapter": None
    }
]

reference_speech = """
Mes chers compatriotes,

La croissance économique n'est pas une fin en soi, mais un moyen essentiel pour améliorer la prospérité de notre nation. Nous devons l'aborder avec pragmatisme et responsabilité. Notre approche repose sur trois piliers fondamentaux : l'innovation, la stabilité fiscale et l'investissement stratégique. Les données montrent clairement qu'une réduction ciblée des charges sur les entreprises génératrices d'emplois peut stimuler l'économie sans compromettre nos services publics.

L'expérience internationale prouve que la flexibilité du marché du travail, accompagnée d'un soutien adéquat à la formation professionnelle, permet d'adapter notre économie aux défis contemporains. Notre politique vise à réduire progressivement le taux de chômage de 7% à 5% sur trois ans, tout en maintenant l'inflation sous la barre des 2%. Ces objectifs sont ambitieux mais réalisables si nous maintenons le cap sur la rigueur budgétaire et l'ouverture commerciale.

La croissance ne doit pas se faire au détriment de notre planète ni creuser les inégalités sociales. C'est pourquoi nous proposons d'orienter les investissements publics vers les secteurs d'avenir comme la transition énergétique et la numérisation de l'économie. Notre plan économique prévoit d'allouer 3% du PIB à la recherche et développement d'ici 2027, contre 2.2% actuellement. Cette vision de long terme, basée sur des fondamentaux solides et des objectifs chiffrés, permettra à notre pays de prospérer dans un monde en constante mutation.
"""

def build_prompt(role, topic, tone, constraints):
    return f"""Tu es un politicien {role} qui s'exprime sur {topic}.
    Adopte un ton {tone}. {constraints}
    Génère un discours structuré et cohérent en 3 paragraphes."""

prompt = build_prompt(
    role="de droite",
    topic="la croissance économique",
    tone="pragmatique",
    constraints="Sois crédible, évite les exagérations, et appuie-toi sur des arguments économiques concrets."
)

def evaluate_quality(generated_text, reference_text):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = scorer.score(generated_text, reference_text)
    
    paragraphs = [p for p in generated_text.split("\n\n") if p.strip()]
    sentences = sent_tokenize(generated_text)
    
    word_count = len(generated_text.split())
    
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
        "paragraphs": len(paragraphs),
        "sentences": len(sentences),
        "words": word_count
    }

def compare_models():
    results = []
    
    for model_info in models_to_evaluate:
        print(f"\n{'='*50}\nÉvaluation de {model_info['name']}...\n{'='*50}")
        
        try:
            start_time = time.time()
            
            tokenizer = AutoTokenizer.from_pretrained(model_info["base_model"])
            base_model = AutoModelForCausalLM.from_pretrained(
                model_info["base_model"],
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            if model_info["lora_adapter"]:
                model = PeftModel.from_pretrained(base_model, model_info["lora_adapter"])
            else:
                model = base_model
                
            loading_time = time.time() - start_time
            print(f"Temps de chargement: {loading_time:.2f} secondes")
            
            generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            gen_start_time = time.time()
            result = generator(
                prompt,
                max_length=400,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=1
            )
            generation_time = time.time() - gen_start_time
            
            generated_text = result[0]["generated_text"]
            generated_text = generated_text.replace(prompt, "").strip()
            
            quality_metrics = evaluate_quality(generated_text, reference_speech)
            
            print(f"Temps de génération: {generation_time:.2f} secondes")
            print(f"\nTexte généré:\n----------\n{generated_text}\n----------")
            
            results.append({
                "Modèle": model_info["name"],
                "Temps de chargement (s)": loading_time,
                "Temps de génération (s)": generation_time,
                "Nombre de mots": quality_metrics["words"],
                "Nombre de paragraphes": quality_metrics["paragraphs"],
                "ROUGE-1": quality_metrics["rouge1"],
                "ROUGE-2": quality_metrics["rouge2"],
                "ROUGE-L": quality_metrics["rougeL"],
                "Texte généré": generated_text
            })
            
        except Exception as e:
            print(f"Erreur avec le modèle {model_info['name']}: {str(e)}")
            results.append({
                "Modèle": model_info["name"],
                "Erreur": str(e)
            })
        
        if 'model' in locals():
            del model
        if 'base_model' in locals():
            del base_model
        if 'generator' in locals():
            del generator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    df_results = pd.DataFrame(results)
    
    df_results.to_csv("model_comparison_results.csv", index=False)
    
    return df_results

def visualize_results(df):
    df_metrics = df.drop(columns=["Texte généré", "Erreur"] if "Erreur" in df.columns else ["Texte généré"])
    
    plt.figure(figsize=(12, 6))
    df_metrics.plot(x="Modèle", y=["Temps de chargement (s)", "Temps de génération (s)"], kind="bar")
    plt.title("Comparaison des temps de traitement par modèle")
    plt.ylabel("Temps (secondes)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison_time.png")
    
    plt.figure(figsize=(12, 6))
    df_metrics.plot(x="Modèle", y=["ROUGE-1", "ROUGE-2", "ROUGE-L"], kind="bar")
    plt.title("Scores de qualité ROUGE par modèle")
    plt.ylabel("Score ROUGE")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison_rouge.png")
    
    plt.figure(figsize=(12, 6))
    df_metrics.plot(x="Modèle", y=["Nombre de mots", "Nombre de paragraphes"], kind="bar", secondary_y=["Nombre de paragraphes"])
    plt.title("Caractéristiques des textes générés")
    plt.ylabel("Nombre de mots")
    plt.right_ylabel("Nombre de paragraphes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("model_comparison_text_features.png")

if __name__ == "__main__":
    print("Démarrage de la comparaison des modèles...")
    results_df = compare_models()
    print("\nRésultats comparatifs:")
    print(results_df.drop(columns=["Texte généré"]))
    
    try:

        visualize_results(results_df)
        print("\nGraphiques sauvegardés!")
    except Exception as e: 
        print(f"Erreur lors de la visualisation: {str(e)}")
        
    print("\nRésultats complets sauvegardés dans 'model_comparison_results.csv'")