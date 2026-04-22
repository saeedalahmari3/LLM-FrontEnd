from ollama import embed, pull, list as ollama_list
import json

def get_phi_embeddings(inputs):
    model_name = 'phi4-mini'
    
    # Check if model exists
    try:
        models_response = ollama_list()
        available_models = [model.get('name') or model.model for model in models_response.get('models', [])]
        model_names = [model.strip(':latest') for model in available_models]
        if not any(model_name in name for name in model_names):
            print(f"Model {model_name} not found. Pulling...")
            pull(model_name)
    except Exception as e:
        print(f"Error checking models: {e}")
        print(f"Attempting to pull {model_name}...")
        pull(model_name)
    
    response = embed(
        model=model_name,
        input=inputs
    )
    return response['embeddings']


#print(get_phi_embeddings('I am Saeed Alahmari'))