import pandas as pd

# Load the TSV file into a DataFrame
file_path = '/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/test.tsv'
df = pd.read_csv(file_path, sep='\t')

# Ensure the necessary columns are present
required_columns = ['Codes', 'Type']
if not all(column in df.columns for column in required_columns):
    raise ValueError(f"The input file must contain the columns: {required_columns}")

# Filter the DataFrame to contain 1000 rows where the column 'Codes' contains the substring "C-SEP" and 'Type' is 'T'
filtered_df = df[(df['Codes'].str.contains('C-SEP', na=False)) & (df['Type'] == 'T')].head(1000)

# Ensure all columns have the correct data types (all non-numeric columns as strings)
for column in filtered_df.columns:
    if filtered_df[column].dtype == object:
        filtered_df[column] = filtered_df[column].astype(str)

# Save the filtered DataFrame to a new TSV file with the suffix '_filtered'
output_file_path = file_path.replace('.tsv', '_filtered.tsv')
filtered_df.to_csv(output_file_path, sep='\t', index=False)

print(f"Filtered TSV file saved to: {output_file_path}")
