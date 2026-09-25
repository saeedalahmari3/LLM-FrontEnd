#runModel_Ollama 

from ollama import chat, pull, list as ollama_list

def get_ollama_model_response(model, prompt, parameter='synonyms'):
    # Check if model is available, pull if not
    models_response = ollama_list()
    available_models = [model.get('name') or model.model for model in models_response.get('models', [])]
    available_models2 = [model.strip(':latest') for model in available_models]
    #print(available_models2)
    if model not in available_models:
        print(f"Model {model} not found. Pulling...")
        pull(model)
    
    if parameter == 'synonyms':
        response = chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={
            "temperature": 0
            }
        )  

    return response.message

#resp = get_ollama_model_response('phi4-mini','Where FIFA world cup 2026 will take place?')
#print(resp['content'])