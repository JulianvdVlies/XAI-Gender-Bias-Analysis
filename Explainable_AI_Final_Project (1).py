import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

professions = [
    "accountant", "architect", "attorney", "chiropractor", "comedian", 
    "composer", "dentist", "dietitian", "dj", "filmmaker", 
    "interior_designer", "journalist", "model", "nurse", "painter", 
    "paralegal", "pastor", "personal_trainer", "photographer", "physician", 
    "poet", "professor", "psychologist", "rapper", "software_engineer", 
    "surgeon", "teacher", "yoga_teacher"
]
print("Loading dataset...")
dataset = load_dataset("LabHC/bias_in_bios", split = "test[:10]")
classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli", device = -1)

def run_shap(text, target_label):
    def score_fn(texts):
        if isinstance(texts, str):
            text_list = [texts]
        else:
            text_list = [str(t) for t in texts]

        results = classifier(text_list, candidate_labels = professions)

        if isinstance(results, dict):
            results = [results]
            
        final_scores = []
        for res in results:
            label_index = res["labels"].index(target_label)
            final_scores.append(res["scores"][label_index])
        return np.array(final_scores)

    explainer = shap.Explainer(score_fn, classifier.tokenizer)
    shap_values = explainer([text], max_evals = 50)

    print(f"\nSHAP visual for label: {target_label}")
    shap.plots.waterfall(shap_values[0])

def plot_attention(text):
    attn_model_name = "distilber-base-uncased" 
    tokenizer = AutoTokenizer.from_pretrained(attn_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(attn_model_name, output_attentions = True)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    attention = outputs.attentions[-1][0].mean(dim = 0)
    cls_attention = attention[0].detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    plt.figure(figsize=(10, 4))
    plt.bar(tokens, cls_attention)
    plt.xticks(rotation=45)
    plt.title("Attention weights (Internal Model Focus)")
    plt.ylabel("Attention Score")
    plt.show()

example_index = 0
raw_text = dataset[example_index]["hard_text"]
true_label_id = dataset[example_index]["profession"]
true_label_name = professions[true_label_id]

print(f"Testing Bio: {raw_text[:100]}...")
print(f"True Profession: {true_label_name}")

run_shap(raw_text, "surgeon")

plot_attention(raw_text)