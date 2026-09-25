#Apply Typo 

import typo 
import random 
import re
random.seed(42)
# Seed for reproducibility
print('*'*30)
""" 
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
"""
def apply_typo(text, percentage):
    if not 0 <= percentage <= 1:
        raise ValueError("percentage must be between 0 and 1")

    # Numeric spans are kept in separate parts.
    parts = re.split(r"(\d+)", text)
    text_indexes = range(0, len(parts), 2)

    total_letters = sum(
        char.isalpha()
        for index in text_indexes
        for char in parts[index]
    )
    number_of_changes = int(percentage * total_letters)

    for _ in range(number_of_changes):
        candidates = [
            index
            for index in text_indexes
            if sum(char.isalpha() for char in parts[index]) >= 2
        ]

        if not candidates:
            break

        weights = [
            sum(char.isalpha() for char in parts[index])
            for index in candidates
        ]
        index = random.choices(candidates, weights=weights, k=1)[0]

        parts[index] = (
            typo.StrErrer(parts[index])
            .char_swap()
            .missing_char()
            .result
        )

    return "".join(parts)

#original = "Hello World! Happy new year 2021. Price: 123.45"
#changed = apply_typo(original, 0.06)

#print(changed)
#assert re.findall(r"\d+", changed) == re.findall(r"\d+", original)
#apply_typo(text,0.1)