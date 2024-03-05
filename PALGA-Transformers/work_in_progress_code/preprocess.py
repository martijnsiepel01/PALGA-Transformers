import pandas as pd
import re
import os

def preprocess_sentence(sentence):
    # Remove dates
    sentence = re.sub(r'\b\d+-\d+-\d+\b', '', sentence)

    # Remove punctuation and special characters
    sentence = re.sub(r'[^\w\s\[\]]', '', sentence)

    # Convert to lowercase
    sentence = sentence.lower()

    # Replace all occurrences of "[crlf]" with a space
    sentence = sentence.replace("[crlf]", " ")

    return sentence

# Specify the path to your TSV file
tsv_file = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/gold_resolved_with_codes.tsv'

# Read the TSV file into a DataFrame
df = pd.read_csv(tsv_file, sep='\t')

df['Conclusie'] = df['Conclusie'].apply(preprocess_sentence)

# Get the directory and filename of the original TSV file
directory = os.path.dirname(tsv_file)
filename = os.path.basename(tsv_file)

# Create the new file path for the processed TSV file
processed_tsv_file = os.path.join(directory, f"{filename}")

# Write the processed DataFrame to the new TSV file
df.to_csv(processed_tsv_file, sep='\t', index=False)
print("done")

# def print_non_az_characters(tsv_file):
#     # Read the TSV file into a DataFrame
#     df = pd.read_csv(tsv_file, sep='\t')

#     # Concatenate all sentences into a single string
#     sentences = ' '.join(str(df['Conclusie']))

#     # Find all non a-z characters using regex
#     non_az_characters = re.findall(r'[^a-zA-Z\s\t\n]', sentences)

#     # Get unique non a-z characters
#     unique_non_az_characters = set(non_az_characters)

#     # Print the unique non a-z characters as a list
#     print(list(unique_non_az_characters))

# # Specify the path to your TSV file
# tsv_file = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/gold_P1.tsv'

# # Call the function to print non a-z characters
# print_non_az_characters(tsv_file)
