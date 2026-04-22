#Apply Typo 

import typo 
import random 
random.seed(42)
# Seed for reproducibility
print('*'*30)

def apply_typo(text,percentage):
    total_char = len(text)
    number_of_char_to_change = percentage * total_char 
    seeds = [random.randint(1,1000) for _ in range(int(number_of_char_to_change))]
    for i in seeds:
        str_errer = typo.StrErrer(text, seed=i)
        # Apply random character swapping (typo) multiple times
        text = str_errer.char_swap().result
        #print(text) # Example output: "Hlelo World! Happy new year 2021."
        # Apply missing character (typo)
        text = str_errer.missing_char().result
        #print(text) # Example output: "Helo World! Happy new year 2021.
    return text
#apply_typo(text,0.1)