import json
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Path to the JSON file
json_file_path = "data/datasets/dataset_combined_result_mark.json"

# Read the JSON file
with open(json_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract question lengths
question_lengths = []

# Iterate through each item in the JSON
for key, item in data.items():
    if "question" in item:
        # Calculate the length of the question
        question_length = len(item["question"])
        question_lengths.append(question_length)

# Print the result
print(f"Number of questions processed: {len(question_lengths)}")
print(f"Question lengths: {question_lengths}")

# Optionally, save the result to a file
with open("data/datasets/question_lengths.txt", 'w') as output_file:
    output_file.write(str(question_lengths))

# Set the style for a cleaner look
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Set font to Times New Roman and increase font sizes
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

# Generate histogram
fig, ax = plt.subplots(figsize=(10, 6))

# Create bins of size 100
min_length = min(question_lengths)
max_length = max(question_lengths)
bin_edges = np.arange(min_length - min_length % 100, max_length + 100, 100)

# Create the histogram with density=True to normalize for KDE comparison
histogram = ax.hist(question_lengths, bins=bin_edges.tolist(), alpha=1.0, color='#2E8B8B', 
                   edgecolor='white', linewidth=0.5, density=True)

# Generate x values for smooth curve (more points than our bins)
x_values = np.linspace(min_length, max_length, 1000)

# Create kernel density estimation for smooth curve
kde = stats.gaussian_kde(question_lengths, bw_method='scott')
density_curve = kde(x_values)

# Plot the KDE curve
ax.plot(x_values, density_curve, color='#2A2E8C', linewidth=2)

# Clean up the plot - remove spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Add labels and title
ax.set_xlabel('Question Length (characters)', fontsize=20)
ax.set_ylabel('Frequency Distribution', fontsize=20)
ax.set_title('Question Length Distribution', fontsize=24, color='#2E8B57')

# Remove the legend as it's not in the example
# plt.legend()

# Add text with statistics in a neater format
stats_text = (f"Total Questions: {len(question_lengths)}\n"
            f"Min Length: {min_length}\n"
            f"Max Length: {max_length}\n"
            f"Average Length: {sum(question_lengths)/len(question_lengths):.2f}\n"
            f"Median Length: {np.median(question_lengths):.2f}")

# Position and display the statistics text with the Times New Roman font
ax.text(0.73, 0.95, stats_text, transform=ax.transAxes, fontsize=16, verticalalignment='top', 
       fontfamily='Times New Roman', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=5))

# Add lighter horizontal grid lines only
ax.yaxis.grid(True, linestyle='-', alpha=0.2)
ax.xaxis.grid(False)  # Turn off vertical grid lines

# Set y-axis to start at 0
ax.set_ylim(bottom=0)

# Adjust spacing to make it cleaner
plt.tight_layout()

# Save the histogram with higher DPI for better quality
plt.savefig('data/images/question_length_histogram.png', dpi=300, bbox_inches='tight')
print("Histogram with density curve saved as 'question_length_histogram.png'")

# Show the histogram (this won't display in terminal, but useful if running in interactive environment)
plt.show() 