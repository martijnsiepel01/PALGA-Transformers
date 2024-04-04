import pandas as pd

# Specify the path to your TSV file
tsv_file = '/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/pretrain/pretrain.tsv'

# Load the TSV file into a DataFrame
df = pd.read_csv(tsv_file, delimiter='\t')

# Find all rows where the Jaar column == 2023 and the Conclusie column has the words "vrij" and "neus"
# filtered_df = df[(df['Jaar'] == 2023) & (df['Conclusie'].str.contains('vrij')) & (df['Conclusie'].str.contains('neus')) & (df['Conclusie'].str.contains('basaalcelcarcinoom'))]
filtered_df = df[(df['Jaar'] == 2023) & (df['Conclusie'].str.contains('vrij')) & (df['Conclusie'].str.contains('neus')) & (df['Conclusie'].str.contains('nevus'))]

print(len(filtered_df))
for row in filtered_df.itertuples():
    print(row.Conclusie)
    print('----')