import random
import requests
#from bs4 import BeautifulSoup
import spacy
import os
import random 
import os
# Load spaCy English model

import en_core_web_sm
nlp = en_core_web_sm.load()
#nlp = spacy.load("en_core_web_sm")

# IMPORTANT: You need a Merriam-Webster Thesaurus API key for this to work.
# 1. Get a free key from: https://dictionaryapi.com/
# 2. Set it as an environment variable named 'MW_THESAURUS_API_KEY'
#    or replace "YOUR_API_KEY_HERE" with your actual key.
with open('../API_Key/MerriamWebster.txt', 'r') as f:
    api_key = f.read().strip()
    os.environ['MW_THESAURUS_API_KEY'] = api_key
MW_THESAURUS_API_KEY = os.environ.get("MW_THESAURUS_API_KEY", "YOUR_API_KEY_HERE")

#print(MW_THESAURUS_API_KEY)

def get_synonyms_thesaurus(word):
    """Get synonyms using Merriam-Webster's Thesaurus API."""
    if not word or MW_THESAURUS_API_KEY == "YOUR_API_KEY_HERE":
        if MW_THESAURUS_API_KEY == "YOUR_API_KEY_HERE":
            print("Warning: Merriam-Webster API key is not set.")
        return []

    url = f"https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={MW_THESAURUS_API_KEY}"
    #print(MW_THESAURUS_API_KEY)

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        if not data or not isinstance(data, list):
            return []

        # If the word is not found, the API may return a list of suggestions (strings).
        # A valid response for a word with synonyms is a list of dictionaries.
        if not isinstance(data[0], dict):
            return []

        all_synonyms = set()
        for entry in data:
            if not isinstance(entry, dict):
                continue
            # The API response can have synonyms in at least two different places.
            # Case 1: in 'meta' -> 'syns'
            for syn_list in entry.get('meta', {}).get('syns', []):
                all_synonyms.update(syn_list)

            # Case 2: in 'def' -> 'sseq' -> 'sense' -> 'syn_list'
            for definition in entry.get('def', []):
                for sseq_group in definition.get('sseq', []):
                    for sseq_item in sseq_group:
                        if sseq_item[0] == 'sense':
                            sense_obj = sseq_item[1]
                            for syn_list_of_dicts in sense_obj.get('syn_list', []):
                                for syn_dict in syn_list_of_dicts:
                                    synonym = syn_dict.get('wd')
                                    if synonym:
                                        all_synonyms.add(synonym)
        
        return list(all_synonyms)[:10]

    except (requests.exceptions.RequestException, ValueError):
        # Covers network errors, timeouts, and JSON decoding errors.
        return []


def extract_random_words(sentence, pos_tags=("VERB"), sample_size=2): #"NOUN",
    """Extract nouns or verbs randomly from a sentence."""
    doc = nlp(sentence)
    candidates = [token.text for token in doc if token.pos_ in pos_tags]

    if not candidates:
        return []

    random.shuffle(candidates)
    return candidates[:sample_size]


def get_random_synonyms(sentence, sample_size=2):
    selected_words = extract_random_words(sentence, sample_size=sample_size)
    result = {}

    for word in selected_words:
        synonyms = get_synonyms_thesaurus(word.lower())
        result[word] = synonyms

    for key in result.keys():
        try:
            indx = random.randint(0,len(result)-1)
            sentence = sentence.replace(key,result[key][indx])
        except Exception as e:
            Warning('Error encountered in updating the sentence {}'.format(e))
    return result, sentence



# -------------------------------
# Example usage
# -------------------------------
#sentence = "the rock is destined to be the 21st century's new  conan" \
#"and that he's going to make a splash even greater than arnold schwarzenegger , " \
#"jean-claud van damme or steven segal ."
#output, sentence = get_random_synonyms(sentence, sample_size=2)
#print(output)


