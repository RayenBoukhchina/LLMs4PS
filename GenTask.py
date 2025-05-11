from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel, PeftConfig  
import torch

#base_model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Alternative plus légère
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float16, device_map="auto")

# Si fine-tuning via LoRA est appliqué, charge le modèle fine-tuné :
model = PeftModel.from_pretrained(model, "TinyLlama/tinyllama-chat-1.1b-v1.0-lora")
 
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

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = generator(
    prompt,
    max_length=400,
    do_sample=True,
    top_p=0.9,         
    temperature=0.7,
    num_return_sequences=1
)

print(result[0]["generated_text"])
