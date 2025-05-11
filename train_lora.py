import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import Trainer, DataCollatorForLanguageModeling

# Utilisation forcée du CPU pour plus de stabilité
device = torch.device("cpu")
print("Utilisation du CPU pour plus de stabilité")

# Charger le modèle de base avec les paramètres adaptés à MPS
base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32,  
    device_map=None             
)

# Déplacer le modèle vers le dispositif
model = model.to(device)

# Configuration LoRA
lora_config = LoraConfig(
    r=8,                      
    lora_alpha=16,           
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Préparation du modèle pour LoRA
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
print(f"Modèle préparé pour LoRA avec {model.num_parameters()} paramètres")

# Charger le dataset CSV
try:
    dataset = load_dataset("csv", data_files="english_politicians.csv")
    
    # Afficher les colonnes disponibles
    print("Colonnes disponibles:", dataset["train"].column_names)
    
    # Utiliser spécifiquement la colonne 'Speech'
    text_column = "Speech"  
    print(f"Utilisation de la colonne '{text_column}' pour les discours")
    
    # Fonction de tokenization pour la colonne 'Speech'
    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=512)

except Exception as e:
    print(f"Erreur lors du chargement du dataset: {e}")
    print("Création d'un petit dataset exemple...")
    
    from datasets import Dataset
    import pandas as pd
    
    sample_data = pd.DataFrame({
        "Speech": [
            "Mes chers compatriotes, la croissance économique est notre priorité absolue...",
            "En tant que représentant du peuple, je m'engage à défendre nos valeurs...",
            "L'éducation et la santé sont les piliers fondamentaux de notre société..."
        ]
    })
    
    dataset = Dataset.from_pandas(sample_data)
    dataset = {"train": dataset}
    text_column = "Speech"
    
    def tokenize_function(examples):
        return tokenizer(examples[text_column], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)

# Limitez le nombre d'exemples d'entraînement
tokenized_dataset = tokenized_dataset.select(range(min(500, len(tokenized_dataset))))
print(f"Dataset limité à {len(tokenized_dataset)} exemples pour économiser la mémoire")


# Configuration d'entraînement optimisée 
training_args = TrainingArguments(
    output_dir="model-lora-finetune",
    per_device_train_batch_size=1,     
    gradient_accumulation_steps=8,     # Augmenter pour compenser
    num_train_epochs=1,               
    learning_rate=2e-4,
    fp16=False,                        
    bf16=False,                        
    save_total_limit=1,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",                  
    optim="adamw_torch", 
    dataloader_pin_memory=False           
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Trainer avec désactivation de pin_memory
data_loader_kwargs = {"pin_memory": False}
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator 
    )

# Entraînement
print("Patience pendant la compilation des opérations...")
print("Démarrage de l'entraînement...")
trainer.train()

# Sauvegarde du modèle
model.save_pretrained("model-lora-finetune")
print("Modèle LoRA sauvegardé dans le dossier 'model-lora-finetune'")