from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# load pre-trained BERT tokenizer/model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# example input
text = "Hugging Face is honestly a game changer."

# reusable function
def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output.squeeze() 
embedding = get_bert_embedding(text, model, tokenizer)

np.save("bert_embedding.npy", embedding.numpy())
print("Saved to bert_embedding.npy")