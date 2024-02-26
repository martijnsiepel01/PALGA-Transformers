import pandas as pd
import re
import csv

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def split_terms(s):
    if pd.isna(s) or s is None or s == '':
        return []
    
    # Regular expression to match words inside quotes or words outside quotes
    pattern = r"'\s*([^']+\s*[^']*)\s*'|(\S+)"
    matches = re.findall(pattern, s)
    # Extracting matched groups and removing empty strings
    return [m[0] if m[0] else m[1] for m in matches]

# Function to normalize text by removing extra spaces, dots at the end, and trimming spaces
def normalize_text(text):
    if pd.isnull(text):
        return ""  # Return an empty string for NaN or None values
    # Remove all non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Trim leading and trailing spaces
    text = text.strip()
    # Convert to lowercase
    text = text.lower()
    return text


def find_codes(terms):
    codes = []
    count_not_found = 0  # Counter for no matching rows
    not_found_terms = []  # List to keep track of terms not found
    count_found = 0  # Counter for matching rows
    
    for term in terms:
        if term == '[C-SEP]':
            codes.append('[C-SEP]')
        else:
            normalized_term = normalize_text(term)
            # First try to match in the 'Normalized_DETEROM' column
            matching_rows = thesaurus[thesaurus['Normalized_DETEROM'] == normalized_term]
            
            if matching_rows.empty:
                # If no match, try in the 'Normalized_DEADVOM' column
                matching_rows = thesaurus[thesaurus['Normalized_DEADVOM'] == normalized_term]
            
            if not matching_rows.empty:
                code = matching_rows.iloc[0]['DEPALCE']
                codes.append(code)
                count_found += 1
            else:
                count_not_found += 1
                not_found_terms.append(term)  # Add the term to not found list
    
    return codes, count_not_found, not_found_terms, count_found

def list_to_string(lst):
    return ' '.join(lst)  # Adjust the separator as needed

# Save the result to a new file
common_paths = ['/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/gold_P1',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_train',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_validation',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_test',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_train',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_validation',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_test',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_train',
                '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_validation']


thesaurus_location = '/home/gburger01/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')
thesaurus['Normalized_DETEROM'] = thesaurus['DETEROM'].apply(normalize_text)
thesaurus['Normalized_DEADVOM'] = thesaurus['DEADVOM'].apply(normalize_text)

# Initialize global accumulators
all_not_found_terms = set()
total_no_match_count = 0
total_count_found = 0

not_found_count = {}

for path in common_paths:
    input_location = path + '.tsv'
    output_location = path + '_with_codes2.tsv'

    input_df = pd.read_csv(input_location, sep='\t')

    # Assuming a function to split 'Codes' into terms is defined as split_terms
    input_df['terms'] = input_df['Codes'].apply(split_terms)

    # Apply the find_codes function to each item in 'terms' column and collect results
    results = input_df['terms'].apply(find_codes)

    # Extract results
    codes = results.apply(lambda x: x[0])
    count_not_found = results.apply(lambda x: x[1])
    not_found_lists = results.apply(lambda x: x[2])
    count_found = results.apply(lambda x: x[3])

    # Assign the codes back to the DataFrame
    input_df['Palga_codes'] = codes.apply(list_to_string)

    # Update global accumulators
    total_no_match_count += count_not_found.sum()
    total_count_found += count_found.sum()
    all_not_found_terms.update(set(term for sublist in not_found_lists for term in sublist))

    # Update not_found_count dictionary
    for sublist in not_found_lists:
        for term in sublist:
            if term not in not_found_count:
                not_found_count[term] = 1
            else:
                not_found_count[term] += 1

    # # Save the modified DataFrame
    # input_df.to_csv(output_location, sep='\t', index=False)

# At this point, all_not_found_terms, total_no_match_count, and total_count_found are accumulated across all files
print("Total terms not found:", len(all_not_found_terms))
print("Total no match count:", total_no_match_count)
print("Total count of found terms:", total_count_found)


# Load the data
df1_path = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/work_in_progress_code/similar_terms_embedding.csv'
df2_path = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/work_in_progress_code/similar_terms_tfidf.csv'

df1 = pd.read_csv(df1_path)
df2 = pd.read_csv(df2_path)

# Ensure that the "Count" column exists in both DataFrames, initialized with zeros or updated if already present
df1['Count'] = 0
df2['Count'] = 0

# Function to update counts in a DataFrame
def update_counts(df, not_found_count):
    # For each term in not_found_count, find it in 'Not_Found_Term' and update 'Count'
    for term, count in not_found_count.items():
        # Check if the term exists in the DataFrame
        mask = df['Not_Found_Term'] == term
        if mask.any():  # If the term is found
            df.loc[mask, 'Count'] = count  # Update the count for the term

# Update counts in both DataFrames
update_counts(df1, not_found_count)
update_counts(df2, not_found_count)

# Save the updated DataFrames back to their original locations
df1.to_csv(df1_path, index=False)
df2.to_csv(df2_path, index=False)

print("Updated CSV files have been saved.")

# # Combine the normalized columns for vectorization
# combined_columns = thesaurus['Normalized_DETEROM'].tolist() + thesaurus['Normalized_DEADVOM'].tolist()

# # Initialize the vectorizer
# vectorizer = TfidfVectorizer()
# # Fit and transform the combined data
# tfidf_matrix = vectorizer.fit_transform(combined_columns)

# # Function to find the most similar term
# def find_most_similar(term, tfidf_matrix, vectorizer, thesaurus):
#     # Vectorize the input term
#     term_vec = vectorizer.transform([term])
#     # Compute cosine similarity
#     cosine_similarities = cosine_similarity(term_vec, tfidf_matrix).flatten()
#     # Find the index of the highest similarity
#     max_sim_index = cosine_similarities.argmax()
#     # Calculate the highest similarity score
#     max_sim_score = cosine_similarities[max_sim_index]
#     # Retrieve the matching term from combined columns
#     if max_sim_index < len(thesaurus['Normalized_DETEROM']):
#         matching_term = thesaurus.iloc[max_sim_index]['Normalized_DETEROM']
#     else:
#         matching_term = thesaurus.iloc[max_sim_index - len(thesaurus['Normalized_DETEROM'])]['Normalized_DEADVOM']
#     return term, matching_term, max_sim_score

# # Initialize a list to hold the result
# similar_terms_list = []

# # Find the most similar term for each not found term
# for not_found_term in all_not_found_terms:
#     similar_terms_list.append(find_most_similar(not_found_term, tfidf_matrix, vectorizer, thesaurus))

# # Convert the list to a DataFrame
# similar_terms_df = pd.DataFrame(similar_terms_list, columns=['Not_Found_Term', 'Found_Term', 'Similarity'])

# similar_terms_df.to_csv('/home/gburger01/PALGA-Transformers/PALGA-Transformers/work_in_progress_code/similar_terms2.csv', index=False)
