import pandas as pd
import re
import csv

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

found_codes_count = 0
missing_codes_count = 0

# Modify the function to count the occurrence of each missing code
def find_missing_codes(codes, missing_codes_dict):
    global found_codes_count
    # print(codes)
    for code in codes:
        if code != '[C-SEP]' and code not in thesaurus['DEPALCE'].values:
            if code in missing_codes_dict:
                missing_codes_dict[code] += 1
            else:
                missing_codes_dict[code] = 1
        else:
            found_codes_count += 1

# Initialize a dictionary to count each missing code
missing_codes_dict = {}

thesaurus_location = '/home/gburger01/snomed_20230426.txt'
thesaurus = pd.read_csv(thesaurus_location, sep='|', encoding='latin-1')

histo_train_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_train_with_codes.tsv"
histo_test_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_test_with_codes.tsv"
histo_validation_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_validation_with_codes.tsv"

all_train_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_train_with_codes.tsv"
all_test_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test_with_codes.tsv"
all_validation_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_validation_with_codes.tsv"

autopsy_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_new.tsv"

paths = [histo_train_path, histo_test_path, histo_validation_path, all_train_path, all_test_path, all_validation_path, autopsy_path]

for path in paths:
    print(path)
    df = pd.read_csv(path, delimiter='\t')
    df['Codes'] = df['Codes'].str.split()
    for index, row in df.iterrows():
        codes = row['Codes']
        if str(codes) != 'nan':
            find_missing_codes(codes, missing_codes_dict)

# Update the missing_codes_count to be the sum of counts of all missing codes
missing_codes_count = sum(missing_codes_dict.values())

sorted_missing_codes = sorted(missing_codes_dict.items(), key=lambda x: x[1], reverse=True)
print(sorted_missing_codes)
print(f"Found Codes Count: {found_codes_count}")   
print(f"Missing Codes Count: {missing_codes_count}")

# ['M75700', 'M89553', 'M90823', 'M97613', 'T56110M81603', 'T56110M81603', 'M89503', 'P11150', 'M90823', 'P11150', 'P11150', 'T56110M81603', 'P11150', 'P11150', 'M89503', 'M89506', 'T56110M81603', 'M97613', 'M88311', 'P11150', 'P11150', 'M89503', 'M97613', 'M96703', 'M97613', 'M90823', 'M90823', 'M97613', 'M75700', 'M89503', 'M88311', 'M97613', 'M89503', 'P11150', 'P11150', 'M89503', 'M88311', 'M75700', 'M96703', 'M91023', 'P11150', 'M89503', 'M88311', 'M90823', 'M89553', 'P11150', 'M88311', 'T56110M81603', 'T56110M81603', 'P11150', 'P11150', 'M89503', 'M90823', 'M97613', 'M90823', 'M84411', 'M86500', 'M84701', 'M75700', 'M82140', 'T56110M81603', 'M96703F40640', 'M92230', 'M89506', 'M89553', 'M89503', 'P11150', 'M97613', 'M89503', 'M90823', 'M91023', 'M88313', 'P11150', 'P11150', 'M87000', 'M97613', 'T56110M81603', 'M89503', 'M89506', 'M89553', 'M96703', 'M96703', 'M90826', 'M97613', 'M96703', 'M89503', 'M89503', 'M90823', 'M96703', 'M96703', 'M97613', 'M90823', 'M90823', 'M82140', 'M89553', 'M90823', 'M89503', 'M89503', 'M90823', 'M96703F40640', 'M96703F40640', 'M89553', 'M90823', 'M89503', 'M88313', 'M96703', 'P11150', 'M91023', 'M90823', 'M90823', 'M89553', 'M88311', 'M89503', 'M86931', 'M90823', 'P11150', 'T56110M81603']
# 120894
# 117