import pandas as pd
from datasets import load_metric
import numpy as np
from scipy.stats import ttest_ind

# Path to the TSV files
file_path_1 = fr'C:\Users\Martijn\OneDrive\Thesis\data\test_with_custom_loss_predictions.tsv'
file_path_2 = fr'C:\Users\Martijn\OneDrive\Thesis\data\test_with_default_loss_predictions.tsv' # Path to the second file

# Read the TSV files into DataFrames
df1 = pd.read_csv(file_path_1, sep='\t')
df2 = pd.read_csv(file_path_2, sep='\t')

# Define length ranges
length_ranges = {
    '0-80': (0, 80),
    '81-200': (81, 200),
    '201-400': (201, 400),
    '400+': (401, float('inf'))
}

bleu_metric = load_metric("bleu")

# Function to filter by length
def filter_by_length(conclusion, min_length, max_length):
    return min_length <= len(conclusion) <= max_length

# Function to calculate BLEU score
def calculate_bleu(df, name):
    if len(df) == 0:
        print(f"No data for {name}")
        return 0
    predictions = [trans.lower().split() for trans in df['Predictions'].tolist()]
    references = [[ref.lower().split()] for ref in df['Codes'].tolist()]

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    return bleu_score['bleu']


# Function to bootstrap dataset
def bootstrap_sample(df, n_iterations=10):
    n_samples = len(df)
    bootstrapped_datasets = [df.sample(n=n_samples, replace=True) for _ in range(n_iterations)]
    return bootstrapped_datasets

# Function to calculate BLEU score with bootstrapping
def calculate_bleu_with_bootstrapping(df1, df2, n_iterations=10):
    groups = {}
    for type_ in df1['Type'].unique():
        type_df1 = df1[df1['Type'] == type_]
        type_df2 = df2[df2['Type'] == type_]
        for length_range, (min_length, max_length) in length_ranges.items():
            group_name = f'{type_}_{length_range}'
            filtered_df1 = type_df1[type_df1['Conclusie_Conclusie'].apply(lambda x: filter_by_length(x, min_length, max_length))]
            filtered_df2 = type_df2[type_df2['Conclusie_Conclusie'].apply(lambda x: filter_by_length(x, min_length, max_length))]

            bootstrapped_datasets1 = bootstrap_sample(filtered_df1, n_iterations)
            bootstrapped_datasets2 = bootstrap_sample(filtered_df2, n_iterations)

            bleu_scores1 = [calculate_bleu(bootstrapped_df, f"{group_name}_bootstrap_{i+1}") for i, bootstrapped_df in enumerate(bootstrapped_datasets1)]
            bleu_scores2 = [calculate_bleu(bootstrapped_df, f"{group_name}_bootstrap_{i+1}") for i, bootstrapped_df in enumerate(bootstrapped_datasets2)]

            avg_bleu_score1 = np.mean(bleu_scores1)
            avg_bleu_score2 = np.mean(bleu_scores2)

            print(f"No bleu scores model 1: {len(bleu_scores1)}")
            print(f"No bleu scores model 2: {len(bleu_scores2)}")
            
            print(f"Number of samples for {group_name} in datasets:", len(filtered_df1))
            print(f"Average BLEU score for {group_name} in dataset 1 after bootstrapping:", avg_bleu_score1)
            print(f"Average BLEU score for {group_name} in dataset 2 after bootstrapping:", avg_bleu_score2)

            # Perform t-test to determine if there is a statistical difference
            t_stat, p_value = ttest_ind(bleu_scores1, bleu_scores2)
            print(f"t-statistic: {t_stat}, p-value: {p_value}")
            print()

calculate_bleu_with_bootstrapping(df1, df2, n_iterations=100)
