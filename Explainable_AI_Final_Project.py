""" 
Bias analysis in professional biographies using SHAP and attention visualization.
Author: Julian van der Vlies - s1141892
Course: Explainable AI
Description: This script evaluates gender bias in a pre-trained DistilBERT model 
by comparing post-hoc SHAP values with intrinsic model attention weights."""
import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings

#Ignore warnings for a more neat output
warnings.filterwarnings("ignore")

model_name = "distilbert-base-uncased"
num_examples = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#list of professions in the dataset, used for mapping label indices to profession names
professions = [
    "accountant", "architect", "attorney", "chiropractor", "comedian", 
    "composer", "dentist", "dietitian", "dj", "filmmaker", 
    "interior_designer", "journalist", "model", "nurse", "painter", 
    "paralegal", "pastor", "personal_trainer", "photographer", "physician", 
    "poet", "professor", "psychologist", "rapper", "software_engineer", 
    "surgeon", "teacher", "yoga_teacher"]

#set of gendered terms, used for highlighting bias in plots
gendered = {"he", "his", "him", "she", "her", "hers", "mr", "ms", "mrs", "mother", "father", "sister", "brother", "wife", "husband"}

#model loading
print(f"Loading model on {device}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Load pre-trained model for sequence classification with attention outputs enabled
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = len(professions), output_attentions = True).to(device)
model.eval()

#dataset loading
print("Loading dataset")
dataset = load_dataset("LabHC/bias_in_bios", split = f"test[:{num_examples}]")

#Helper function 1: prediction function for SHAP
#Gets class probabilities and returns a numpy array of softmax probabilities.
def predict(texts):
    if isinstance(texts, str):
        texts = [texts]
    enc = tokenizer(list(texts), return_tensors = "pt", truncation = True, padding = True, max_length = 512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim = -1).cpu().numpy()

#Helper function 2
#Extracts attention weights from the last layer of the model.
#Averages weights across all heads for the [CLS] token
def get_attention(text):
    enc = tokenizer(text, return_tensors = "pt", truncation = True, max_length = 512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    attention = out.attentions[-1][0].mean(0)[0].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return tokens, attention

#Creates side by side SHAP and attention plots for comparison.
def plot_example(text, true_label, idx):
    probs = predict(text)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = professions[pred_idx]

    #calculate SHAP values (External focus)
    explainer = shap.Explainer(predict, tokenizer)
    shap_values = explainer([text], max_evals = "auto")
    shap_tokens = list(shap_values.data[0])
    values = shap_values[0, :, pred_idx].values
    #get top 10 most impactful tokens for SHAP
    top = np.argsort(np.abs(values))[-10:][::-1]

    #Process attention weights (Internal focus)
    tokens, attention = get_attention(text)
    #filters out padding and special tokens for cleaner visual
    clean = [(t, a) for t, a in zip(tokens, attention) if t not in ("[CLS]", "[SEP]", "[PAD]")]

    if not clean:
        print(f"skipping example {idx + 1} due to tokenization issues")
        return

    attn_tok, attn_vals = zip(*clean)

    #Get top 10 tokens with highest attention score
    top_a = np.argsort(attn_vals)[-10:][::-1]

    #Create side by side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 5))
    fig.suptitle(f"Example {idx + 1} | True: {true_label} | Predicted {pred_label}" f"({probs[pred_idx]:.1%})", fontweight = "bold")

    #SHAP plot, Red = supports prediction, blue = opposes prediction
    colors_s = ["red" if v > 0 else "blue" for v in values[top]]
    ax1.barh([shap_tokens[i] for i in top][::-1], values[top][::-1], color=colors_s[::-1])
    ax1.axvline(0, color="black", lw=0.8)
    ax1.set_title("SHAP (red = toward predicted class)")
    ax1.set_xlabel("SHAP value")
    
    #Attention plot, Red = gendered word, Blue = professional/other word 
    attn_colors = ["red" if t.lower() in gendered else "blue" for t in [attn_tok[i] for i in top_a]]
    ax2.bar([attn_tok[i] for i in top_a], [attn_vals[i] for i in top_a], color=attn_colors)
    ax2.set_xticklabels([attn_tok[i] for i in top_a], rotation=45, ha="right")
    ax2.set_title("Attention (red = gendered word)")
    ax2.set_ylabel("Attention score")    

    plt.tight_layout()
    #Save the figure in the same folder
    plt.savefig(f"example_{idx + 1}_{true_label}.png", dpi = 150, bbox_inches = "tight")
    plt.show()
    plt.close()

#Main loop
for i, sample in enumerate(dataset):
    text = sample["hard_text"]
    true_label = professions[sample["profession"]]
    print(f"\n[{i + 1} / {num_examples}] True Label: {true_label}")
    print(f"Text snippet: {text[:100]}...")
    plot_example(text, true_label, i)