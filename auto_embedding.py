from transformers import AutoTokenizer, AutoModel
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = mean_pooling(outputs, inputs['attention_mask'])

    return embedding.squeeze()

sample_text = "JDRE is going to publish an amazing paper."

models = {
    "SciBERT": "allenai/scibert_scivocab_uncased",
    "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
    "PubMedBERT": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "BioLinkBERT": "kamalkraj/BioSimCSE-BioLinkBERT-BASE",
    "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "SapBERT": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR" # support 50+ langauges
}

for name, model_id in models.items():
    print(f"\n{name} embedding:")
    embedding = get_embedding(sample_text, model_id)
    print(embedding[:6]) 