import pandas as pd
import numpy as np
import random


# Specify the paths to your TSV files
file_paths = [
    "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_validation_with_codes.tsv",
    "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_validation.tsv",
    "/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_validation_with_codes.tsv"
]

# Load the TSV files into a DataFrame
dfs = []
for file_path in file_paths:
    df = pd.read_csv(file_path, sep="\t")
    dfs.append(df)

# Concatenate the DataFrames
df = pd.concat(dfs, ignore_index=True)

conclusies = df['Conclusie']
codes = df['Codes']

def span_corruption(sentence, mean_span_length=3, mask_rate=0.15, seed=None, start_extra_id=0):
    """
    Corrupts a sentence by masking spans of words with special tokens and ensures the output sequence
    ends with an <extra_id_x> to mark the end. It returns the corrupted sentence, the target sentence containing
    the masked spans, and the next extra_id index.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    words = sentence.split()
    num_words = len(words)  
    spans = []
    target_spans = []
    i = 0
    extra_id_counter = start_extra_id

    while i < num_words:
        if random.random() < mask_rate:
            span_length = np.random.poisson(mean_span_length)
            span_length = max(1, span_length)  # Ensure at least one word is masked
            span_end = min(i + span_length, num_words)

            spans.append(f"<extra_id_{extra_id_counter}>")
            target_spans.append(f"<extra_id_{extra_id_counter}> {' '.join(words[i:span_end])}")

            i = span_end
            extra_id_counter += 1
        else:
            spans.append(words[i])
            i += 1

    # Append an additional <extra_id_x> at the end of target_spans to mark the end.
    target_spans.append(f"<extra_id_{extra_id_counter}>")
    extra_id_counter += 1  # Increment to account for the appended extra_id at the end

    corrupted_sentence = " ".join(spans)
    target_sentence = " ".join(target_spans)
    
    return corrupted_sentence, target_sentence, extra_id_counter

def translation_pair_span_corruption(source_sentence, target_sentence, mean_span_length=3, mask_rate=0.15, seed=None):
    """
    Corrupts both source and target sentences by masking spans of words, then concatenates them.
    Returns the concatenated corrupted sentences as model input and concatenated target spans as model output.
    """
    # Corrupt the source sentence, starting with extra_id_0
    corrupted_source, target_source, next_extra_id = span_corruption(source_sentence, mean_span_length, mask_rate, seed, 0)
    # Corrupt the target sentence, continuing the extra_id count
    corrupted_target, target_target, _ = span_corruption(target_sentence, mean_span_length, mask_rate, seed, next_extra_id)

    # Concatenate corrupted sentences to form the model's input
    model_input = corrupted_source + " " + corrupted_target
    # Concatenate target spans to form the model's output
    model_output = target_source + " " + target_target
    
    return model_input, model_output

def source_only_span_corruption_with_target_concat(source_sentence, target_sentence, mean_span_length=3, mask_rate=0.15, seed=None):
    """
    Corrupts the source sentence by masking spans of words, then concatenates this corrupted source with the unaltered target sentence.
    Returns the concatenated sentences as model input and the target spans from the source sentence as model output.
    """
    # Corrupt the source sentence
    corrupted_source, target_source, _ = span_corruption(source_sentence, mean_span_length, mask_rate, seed, 0)

    # Concatenate corrupted source sentence with the unaltered target sentence
    model_input = corrupted_source + " " + target_sentence
    # The model output is just the masked spans from the source sentence
    model_output = target_source
    
    return model_input, model_output

headers = [
    "source_sentence",
    "target_sentence",
    "input_sequence_span_corruption",
    "output_sequence_span_corruption",
    "input_sequence_translation_pair_span_corruption",
    "output_sequence_translation_pair_span_corruption",
    "input_sequence_source_only_span_corruption_with_target_concat",
    "output_sequence_source_only_span_corruption_with_target_concat"
]

# Prepare a list to hold all rows of data
data = []

for index, row in df.iterrows():
    source_sentence = row['Conclusie']
    target_sentence = row['Codes']
    if str(target_sentence) == "nan":
        continue
    input_sequence_span_corruption, output_sequence_span_corruption, _ = span_corruption(source_sentence, seed=1)
    input_sequence_translation_pair_span_corruption, output_sequence_translation_pair_span_corruption = translation_pair_span_corruption(source_sentence, target_sentence, seed=1)
    input_sequence_source_only_span_corruption_with_target_concat, output_sequence_source_only_span_corruption_with_target_concat = source_only_span_corruption_with_target_concat(source_sentence, target_sentence, seed=1)
    
    # Collect the variables into a list
    row_data = [
        source_sentence,
        target_sentence,
        input_sequence_span_corruption,
        output_sequence_span_corruption,
        input_sequence_translation_pair_span_corruption,
        output_sequence_translation_pair_span_corruption,
        input_sequence_source_only_span_corruption_with_target_concat,
        output_sequence_source_only_span_corruption_with_target_concat
    ]
    
    # Add the collected data to the overall data list
    data.append(row_data)

# Convert the collected data into a DataFrame
data_df = pd.DataFrame(data, columns=headers)

# Write the DataFrame to a TSV file
data_df.to_csv("/home/msiepel/PALGA-Transformers/PALGA-Transformers/data/pretrain_all/pretrain_all_validation_combined.tsv", sep='\t', index=False)
