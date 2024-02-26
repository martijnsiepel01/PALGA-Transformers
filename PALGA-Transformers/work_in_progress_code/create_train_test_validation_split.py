import pandas as pd
from sklearn.model_selection import train_test_split

# Read the TSV file into a DataFrame
df = pd.read_csv('/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/mixedn_new.tsv', sep='\t')

# First split: Split the data into 90% train+val and 10% test
train_val_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Second split: Split the remaining data into 80% train and 20% val which is 10% of the original dataset
train_df, val_df = train_test_split(train_val_df, test_size=1/9, random_state=42)

# Define base path for writing files
base_path = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_'

# Write the splits to files
train_df.to_csv(f'{base_path}train_with_codes.tsv', sep='\t', index=False)
test_df.to_csv(f'{base_path}test_with_codes.tsv', sep='\t', index=False)
val_df.to_csv(f'{base_path}validation_with_codes.tsv', sep='\t', index=False)

# Print confirmation
print("Splits have been written to files successfully.")
