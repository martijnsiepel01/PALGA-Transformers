import pandas as pd
from sklearn.model_selection import train_test_split

# Read the TSV file into a DataFrame
df = pd.read_csv('/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/dataset_all_raw_new.tsv', sep='\t')

# Remove the first and third columns
df = df.drop(df.columns[[0, 2]], axis=1)

# Rename the remaining columns
df.columns = ['Type', 'Conclusie', 'Codes']

# Remove all enters in the text for a [CRLF] character
df['Conclusie'] = df['Conclusie'].str.replace('\n', '[CRLF]')

# First split: Split the data into 90% train+val and 10% test
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Second split: Split the remaining data into 80% train and 20% val which is 10% of the original dataset
train_df, val_df = train_test_split(train_val_df, test_size=1/9, random_state=42)

# Remove the 'Type' column from train and validation DataFrames
train_df = train_df.drop('Type', axis=1)
val_df = val_df.drop('Type', axis=1)

# Define base path for writing files
base_path = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_'

# Write the splits to files
train_df.to_csv(f'{base_path}train_with_codes.tsv', sep='\t', index=False)
test_df.to_csv(f'{base_path}test_with_codes.tsv', sep='\t', index=False)
val_df.to_csv(f'{base_path}validation_with_codes.tsv', sep='\t', index=False)

# Print confirmation
print("Splits have been written to files successfully.")
