# LLMs4PS: Génération de Discours Politiques avec des LLMs Open-Source

## Description

LLMs4PS (Large Language Models for Political Speech) est un projet qui exploite les modèles de langage open-source pour générer des discours politiques personnalisables. Grâce à une approche modulaire, l'outil permet de créer des allocutions avec différentes orientations politiques, thèmes et tons.

## Fonctionnalités

- Génération de discours politiques structurés en paragraphes
- Personnalisation de l'orientation politique (droite, gauche, centre)
- Adaptation du ton (pragmatique, émotionnel, académique, etc.)
- Paramétrage des contraintes de style et d'argumentation
- Possibilité de fine-tuner le modèle sur des exemples spécifiques

## Installation

```bash
# Créer un environnement virtuel
python -m venv myenv
source myenv/bin/activate  # Sur Mac/Linux
# ou myenv\Scripts\activate  # Sur Windows

# Installer les dépendances
pip install torch torchvision torchaudio transformers peft accelerate sentencepiece

```

## Structure du Projet

- **GenTask.py**: Script principal pour générer des discours politiques
- **train_lora.py**: Script pour fine-tuner le modèle avec LoRA sur des exemples personnalisés

## Utilisation Rapide

Pour générer immédiatement un discours politique:

```bash
python GenTask.py

```
Ce script utilise par défaut TinyLlama (1.1B) avec un adaptateur LoRA pré-entraîné pour produire un discours politique de droite sur la croissance économique.

## Utilisation des Modèles Open-Source
Le projet privilégie les modèles de langage open-source accessibles sans contrainte:


```bash
python

model = PeftModel.from_pretrained(model, "TinyLlama/tinyllama-chat-1.1b-v1.0-lora")
```
Cette approche permet:

Une utilisation sans compte HuggingFace ni clé d'API
Une exécution locale pour préserver la confidentialité
Une personnalisation complète du modèle selon vos besoins

## Personnalisation
Pour personnaliser le discours généré, modifiez ces paramètres dans GenTask.py:

```bash
prompt = build_prompt(
    role="de droite",  # Orientation politique: "de droite", "de gauche", "centriste"
    topic="la croissance économique",  # Sujet du discours
    tone="pragmatique",  # Ton: "pragmatique", "émotionnel", "académique"
    constraints="Sois crédible, évite les exagérations, et appuie-toi sur des arguments économiques concrets."
)
```
## Fine-tuning avec LoRA
Pour adapter le modèle à votre propre style ou corpus:

Préparez votre dataset dans un fichier CSV avec une colonne "Speech"
Exécutez le script de fine-tuning (optimisé pour Mac/CPU)

```bash
python train_lora.py
```
Utilisez votre modèle fine-tuné:
```bash
model = PeftModel.from_pretrained(model, "model-lora-finetune")
```

