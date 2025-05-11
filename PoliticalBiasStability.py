from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import random 

# ------------------------
# Configuration initiale
# ------------------------
#MODEL_NAME = "EleutherAI/gpt-j-6B" 
#MODEL_NAME = "EleutherAI/gpt-neo-2.7B"  # Modèle plus léger 
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"  # Modèle encore plus léger
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer , truncation=True)

# ------------------------
# Énoncé politique et paraphrases
# ------------------------
statement = "Same-sex marriage should be legal."
paraphrases = [
    "Marriage equality is a human right.",
    "Gay couples deserve the same marriage rights as straight couples.",
    "There should be no law against same-sex marriage.",
    "Everyone should be allowed to marry who they love."
]

# ------------------------
# Variations de prompt
# ------------------------
prompts = [
    "Please respond to the following statement: \"{}\"\nYour response:",
    "Do you agree or disagree with the following idea? \"{}\"\nAnswer:",
    "What's your stance on this? \"{}\"\nResponse:",
    "Give your opinion on this: \"{}\"\n--",
]

# ------------------------
# Fonction d'analyse de stance simplifiée
# ------------------------
def interpret_stance(output):
    output = output.lower()
    if any(word in output for word in ["agree", "support", "yes", "should be legal"]):
        return "AGREE"
    elif any(word in output for word in ["disagree", "oppose", "no", "should be illegal"]):
        return "DISAGREE"
    return "NEUTRAL"

# ------------------------
# Mesure de la stabilité
# ------------------------
def stability_test(model_name, inputs):
    stances = []
    for text in tqdm(inputs):
        out = text_generator(text, max_length=100, do_sample=True, temperature=0.7)[0]['generated_text']        
        stance = interpret_stance(out[len(text):])
        stances.append(stance)
    return stances

# ------------------------
# Exécution
# ------------------------
# Test stabilité face à paraphrases
paraphrased_inputs = [prompts[0].format(p) for p in paraphrases]
stances_para = stability_test(MODEL_NAME, paraphrased_inputs)

# Test stabilité face à changements de prompts
prompt_inputs = [p.format(statement) for p in prompts]
stances_prompt = stability_test(MODEL_NAME, prompt_inputs)

# ------------------------
# Visualisation
# ------------------------
def plot_stability(labels, stances, title):
    unique = sorted(set(stances))
    counts = [stances.count(s) for s in unique]
    plt.figure(figsize=(6, 4))
    sns.barplot(x=unique, y=counts, palette="Set2")
    plt.title(title)
    plt.ylabel("Nombre de réponses")
    plt.xlabel("Stance")
    plt.show()

plot_stability(paraphrases, stances_para, "Stabilité aux paraphrases")
plot_stability(prompts, stances_prompt, "Stabilité aux reformulations de prompt")
