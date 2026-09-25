

from together import Together
from openai import OpenAI
from google import genai
from google.genai import types
import math
import os
from runModel_Ollama import *
#client = Together() # auth defaults to os.environ.get("TOGETHER_API_KEY")


with open('../API_Key/openai.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ['OPENAI_API_KEY'] = api_key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

with open('../API_Key/together.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ['TOGETHER_API_KEY'] = api_key
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "YOUR_API_KEY_HERE")

with open('../API_Key/gemini.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ['GEMINI_API_KEY'] = api_key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")

def span_after_message(content):
    """
    Extract tokens and their probabilities that appear right after <|message|>
    until the next control token (like <|...|>).
    """
    last_idx = None
    for i, item in enumerate(content):
        if item['token'] == "<|message|>":
            last_idx = i

    if last_idx is None or last_idx + 1 >= len(content):
        return []

    results = []
    for item in content[last_idx + 1:]:
        token = item.get("token", "")
        logprob = item.get("logprob", None)

        # stop if we reach another control token like <|...|>
        if token.startswith("<|") and token.endswith("|>"):
            break

        # compute probability (if logprob is valid)
        prob = math.exp(logprob) if logprob is not None else None
        results.append({
            "token": token,
            "logprob": logprob,
            "probability": prob
        })

    return results


def translateSentence(sentence, prompt_task, model_name="openai/gpt-oss-20b"):
    """Translate a sentence to Arabic using an LLM model."""
    client = Together()
    if not sentence or not model_name:
        return ""
    
    try:
        # Prepare the prompt for translation
        #Translate the following English sentence to Arabic:
        prompt = f"{prompt_task}\n\n{sentence}"
        
        # Call the model for translation
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=256
        )
        
        # Extract the translated text
        if isinstance(response, str):
            translated_text = response.strip()
        elif isinstance(response, dict) and 'text' in response:
            translated_text = response['text'].strip()
        elif hasattr(response, 'choices') and len(response.choices) > 0:
            translated_text = response.choices[0].message.content.strip()
        elif hasattr(response, 'text'):
            translated_text = response.text.strip()
        else:
            translated_text = str(response).strip()
        
        return translated_text
    
    except Exception as e:
        print(f"Error in translation: {e}")
        return ""

def run_model(prompt, temp = 1, topk = 1, parameter = 'temp', model_name = "openai/gpt-oss-20b"):
    if model_name in ['gpt-5','gpt-4o','gpt-5-nano','o4-mini-2025-04-16']:
         client = OpenAI()
         #pass
    elif model_name in ["openai/gpt-oss-20b","google/gemma-4-31B-it","ssalahmari/google/gemma-3-12b-it-fcdf3056","google/gemma-3n-E4B-it",'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo','mistralai/Ministral-3-14B-Instruct-2512']:
         client = Together()
    elif 'gemini' in model_name:
         client = genai.Client()
         #pass
    elif model_name in ['gemma3:4b','gemma3:12b','phi4:latest','phi4-mini','wao/phi4-mini-instruct:latest','ministral-3:3b']:
         #HERE call ollama 
         return get_ollama_model_response(model_name, prompt, parameter.lower())

    if parameter.lower() == "temp":
        if temp != None:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ],
                temperature=temp,
                logprobs=True
            )
        else:
                response = client.chat.completions.create(
                model=model_name,
                messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ], # use default temperature. 
                logprobs=True
            )
    elif parameter.lower() == 'topk':
        if topk != None:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ],
                top_k=topk,
                logprobs=True
            )
        else:
                response = client.chat.completions.create(
                model=model_name,
                messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ], # use default temperature and top_k. 
                logprobs=True
            )
    elif parameter == 'synonyms':
        # try:
        response = client.chat.completions.create(
                model=model_name,
                messages=[
                {
                    "role": "user",
                    "content": prompt
                }
                ],
                temperature=0
            )
        # except:
        #     response = client.models.generate_content(
        #         model=model_name,
        #         contents=prompt,
        #         config=types.GenerateContentConfig(
        #         temperature=0  # Adjust between 0.0 (deterministic) and 2.0 (creative)
        #         )
        #     )
    #print(response)
    #print(response.choices[0].logprobs)
    # try:
    return {
        'content': response.choices[0].message.content
    }
    # except:
    #     return {
    #         'content': response.text 
    #     }