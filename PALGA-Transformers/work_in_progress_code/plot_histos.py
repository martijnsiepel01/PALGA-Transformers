import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

all_train_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_train_with_codes.tsv"
all_test_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_test_with_codes.tsv"
all_validation_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/all/all_norm_validation_with_codes.tsv"

autopsies_train_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_train.tsv"
autopsies_test_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_test.tsv"
autopsies_validation_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/autopsies/autopsies_norm_validation.tsv"

histo_train_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_train_with_codes.tsv"
histo_test_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_test_with_codes.tsv"
histo_validation_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/histo/histo_norm_validation_with_codes.tsv"

gold_path = "/home/gburger01/PALGA-Transformers/PALGA-Transformers/data/gold_resolved_with_codes.tsv"

train_paths = [all_train_path, autopsies_train_path, histo_train_path]
test_paths = [all_test_path, autopsies_test_path, histo_test_path]
validation_paths = [all_validation_path, autopsies_validation_path, histo_validation_path]
gold_paths = [gold_path]

conclusie_lengths = []

for path in test_paths:
    df = pd.read_csv(path, delimiter='\t')
    conclusie_lengths.extend(df['Conclusie'].str.len())

# Determine the threshold for outliers
threshold = np.percentile(conclusie_lengths, 95)

# Cap values at this threshold
capped_lengths = [min(x, threshold) for x in conclusie_lengths]

# Calculate the percentiles of the capped data
percentiles = np.percentile(capped_lengths, [20, 40, 60, 80])

# Increase the figure size for better visibility of x-axis labels
plt.figure(figsize=(10, 6))

# Plot the histogram with outliers collapsed into a single bin
plt.hist(capped_lengths, bins=100)
plt.xlabel('Length of Conclusie (capped at 95th percentile)')
plt.ylabel('Frequency')
plt.title('Histogram of Conclusie Lengths with Outliers Collapsed')

# Set the maximum number of x-axis ticks with MaxNLocator
plt.gca().xaxis.set_major_locator(MaxNLocator(prune='both', nbins=10))

# Plot the vertical lines for each percentile
for percentile in percentiles:
    plt.axvline(x=percentile, color='r', linestyle='--')

# Calculate every 500th value within the range of capped_lengths
max_length = int(max(capped_lengths))
every_500th_value = np.arange(0, max_length, 500)

# Combine percentile and every 500th value ticks, ensuring uniqueness and sorted order
all_special_ticks = np.sort(np.unique(np.concatenate((percentiles, every_500th_value))))

# Get current x-axis tick locations
current_ticks = plt.gca().get_xticks()

# Combine current ticks with all special ticks
new_ticks = np.sort(np.unique(np.concatenate((current_ticks, all_special_ticks))))

# Create labels for the new ticks
new_labels = ['' for _ in new_ticks]
for idx, tick in enumerate(new_ticks):
    if tick in percentiles:
        new_labels[idx] = f'{tick:.1f}'  # Label for percentiles
    elif tick in every_500th_value:
        new_labels[idx] = str(int(tick))  # Label for every 500th value

# Set the new ticks and labels on the x-axis
plt.xticks(new_ticks, new_labels, rotation=90)

# Lower the plot to make space for the labels
plt.subplots_adjust(bottom=0.2)

# Save the plot
plt.savefig('test_lengths_histogram.png')

plt.clf()
