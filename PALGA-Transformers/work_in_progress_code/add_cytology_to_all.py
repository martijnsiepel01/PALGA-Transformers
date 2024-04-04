import pandas as pd

# File paths
file1_path = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/all.tsv'
file2_path = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/pretrain/pretrain.tsv'

# Load the TSV files
df1 = pd.read_csv(file1_path, delimiter='\t')
df2 = pd.read_csv(file2_path, delimiter='\t')

# Print the unique counts for column Type in df1
print(df1['Type'].value_counts())
# Add 40000 rows from df2 to df1
df1 = pd.concat([df1, df2[df2['Type'] == 'C'].head(40000)], ignore_index=True)
print(df1['Type'].value_counts())

# Save df1 to a new TSV file
output_file_path = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/all_with_50kC.tsv'
df1.to_csv(output_file_path, sep='\t', index=False)