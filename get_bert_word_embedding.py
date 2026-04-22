import torch
from transformers import BertTokenizer, BertModel


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def get_sentence_embedding(sentence):
    # Tokenize and encode the input sentence
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    # Pass tokens through the BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the hidden states from the last layer
    hidden_states = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]
    
    # Use the [CLS] token embedding as the sentence embedding
    cls_embedding = hidden_states[:, 0, :]  # Shape: [batch_size, hidden_size]
    return cls_embedding.squeeze().numpy()

def get_sentence_transformer_embedding(sentence,model):
    embedding = model.encode(sentence,show_progress_bar=True)
    return embedding
    