import json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import numpy as np

def analyze_json_data(file_path):
    # Read the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Initialize counters
    types_counter = Counter()
    problems_counter = Counter()
    
    # Count occurrences of each type and problem
    for key, item in data.items():
        if isinstance(item, dict) and 'type' in item and 'problem' in item:
            types_counter[item['type']] += 1
            problems_counter[item['problem']] += 1
    
    return types_counter, problems_counter

def print_counts(counter, label):
    print(f"\n{label} Distribution:")
    print("-" * 50)
    for item, count in counter.most_common():
        print(f"{item}: {count}")
    print(f"Total unique {label.lower()}: {len(counter)}")
    print(f"Total count: {sum(counter.values())}")

def plot_horizontal_bar(counter, title, filename):
    # Get the data sorted by count (highest to lowest)
    items_sorted = counter.most_common()
    
    # Reverse the order so larger values appear at the top when plotted
    items_sorted.reverse()
    
    # Unpack the sorted items
    labels, values = zip(*items_sorted)
    
    # Calculate percentage for each item
    total = sum(values)
    percentages = [(v/total)*100 for v in values]
    
    # Set font to Times New Roman for the entire plot
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14  # Base font size
    
    # Set up the plot with a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size
    
    # Create a teal color map similar to the example
    # cmap = plt.colormaps.get('GnBu')
    teal_colors = '#2E8B8B'
    
    # Create horizontal bars
    bars = ax.barh(labels, percentages, color=teal_colors, height=0.7, edgecolor='none')
    
    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add percentage labels at the end of each bar
    for i, (bar, percentage) in enumerate(zip(bars, percentages)):
        ax.text(
            percentage + 0.5,  # Slight offset from the end of the bar
            bar.get_y() + bar.get_height()/2,
            f"{percentage:.1f}%",
            va='center',
            ha='left',
            fontsize=28,  # Increased font size further
            color='#444444',
            family='Times New Roman',
            fontweight='normal'
        )
    
    # Set the title with the specified formatting
    ax.set_title(title, fontsize=26, color='#2a7f90', pad=20, loc='left', family='Times New Roman')
    
    # Set x-axis label
    ax.set_xlabel('Percentage of Types', fontsize=28, color='#444444', family='Times New Roman')
    
    # Set y-axis tick font size
    ax.tick_params(axis='y', which='both', left=False, labelsize=30)
    ax.tick_params(axis='x', labelsize=22)
    
    # Explicitly set font for y-axis tick labels to Times New Roman
    for tick in ax.get_yticklabels():
        tick.set_fontname('Times New Roman')
    
    # Explicitly set font for x-axis tick labels to Times New Roman
    for tick in ax.get_xticklabels():
        tick.set_fontname('Times New Roman')
    
    # Set x-axis grid to light gray and only on major ticks
    ax.grid(axis='x', color='#EEEEEE', linestyle='-', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', visible=False)
    
    # Set axis limits with a bit of padding
    ax.set_xlim(0, max(percentages) * 1.15)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure with a white background
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Chart saved as '{filename}'")
    
    plt.close(fig)

def plot_pie_chart(counter, title, filename):
    # Get the data sorted by count
    labels, values = zip(*counter.most_common())
    
    # Calculate percentage for each item
    total = sum(values)
    percentages = [(v/total)*100 for v in values]
    
    # Set font to Times New Roman for the entire plot
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Set up the figure with a clean style
    plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.subplot(111)
    
    # Define specific colors for different problem types
    # Using teal/blue for the first two and deep red for mixed integer programming
    # Create a dictionary to map specific labels to specific colors
    color_mapping = {
        'Linear programming': '#2E8B8B',      # Teal
        'Integer programming': '#2E318B',     # Deep blue
        'Mixed integer programming': '#8B1A1A'  # Deep red
    }
    
    # Map the colors to the actual labels in the data
    colors = [color_mapping.get(label, '#999999') for label in labels]
    
    # Create the pie chart as a donut - capture only the wedges
    wedges = ax.pie(
        values, 
        colors=colors,
        startangle=90, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
        labels=None,
        autopct=None
    )[0]
    
    # Draw a white circle at the center to create a donut chart
    centre_circle = plt.Circle((0, 0), 0.5, fc='white', edgecolor='none')
    ax.add_patch(centre_circle)
    
    # Add the total count in the center
    ax.text(0, 0, f"n = {total}", ha='center', va='center', fontsize=20, family='Times New Roman')
    
    # Add labels and percentages
    for i, (wedge, label, pct) in enumerate(zip(wedges, labels, percentages)):
        # Get angle and radius
        ang = (wedge.theta2 - wedge.theta1)/2. + wedge.theta1
        x = np.cos(np.deg2rad(ang))
        y = np.sin(np.deg2rad(ang))
        
        # Determine horizontal alignment based on angle
        ha = 'left' if x >= 0 else 'right'
        
        # Get connection point on the wedge edge
        conn_x = 0.75 * x
        conn_y = 0.75 * y
        
        # Text position
        text_x = 1.2 * x
        text_y = 1.2 * y
        
        # Draw connecting line
        ax.plot([conn_x, text_x], [conn_y, text_y], color='gray', linewidth=0.8)
        
        # Add label with larger font size
        ax.text(text_x, text_y, label, ha=ha, va='center', fontsize=15, color='#444444', family='Times New Roman')
        
        # Add percentage below label with larger font size
        ax.text(text_x, text_y - 0.15, f"{pct:.1f}%", ha=ha, va='center', fontsize=18, color='#666666', family='Times New Roman')
    
    # Set title with minimal styling
    ax.set_title(title, fontsize=22, color='#444444', pad=20, loc='center', y=1.05, family='Times New Roman')
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Remove all spines and ticks
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as '{filename}'")
    
    plt.close()

def plot_pie_chart_no_text(counter, filename):
    # Get the data sorted by count
    labels, values = zip(*counter.most_common())
    
    # Set font to Times New Roman for the entire plot
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Set up the figure with a clean style
    plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.subplot(111)
    
    # Define specific colors for different problem types
    # Using teal/blue for the first two and deep red for mixed integer programming
    # Create a dictionary to map specific labels to specific colors
    color_mapping = {
        'Linear programming': '#2E8B8B',      # Teal
        'Integer programming': '#2E318B',     # Deep blue
        'Mixed integer programming': '#8B1A1A'  # Deep red
    }
    
    # Map the colors to the actual labels in the data
    colors = [color_mapping.get(label, '#999999') for label in labels]
    
    # Create the pie chart as a donut - capture only the wedges
    ax.pie(
        values, 
        colors=colors,
        startangle=90, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
        labels=None,
        autopct=None
    )
    
    # Draw a white circle at the center to create a donut chart
    centre_circle = plt.Circle((0, 0), 0.5, fc='white', edgecolor='none')
    ax.add_patch(centre_circle)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Remove all spines and ticks
    ax.set_frame_on(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as '{filename}'")
    
    plt.close()

def main():
    file_path = 'data/datasets/dataset_combined_result_mark.json'
    types_counter, problems_counter = analyze_json_data(file_path)
    
    # Print counts
    print_counts(types_counter, "Type")
    print_counts(problems_counter, "Problem")
    
    # Create and save bar chart for Type Distribution
    plot_horizontal_bar(types_counter, 'Percentage of Cases by Problem Type', 'data/images/types_distribution.png')
    
    # Create and save pie chart for Problem Distribution
    plot_pie_chart(problems_counter, 'Problem Distribution', 'data/images/problems_distribution_pie.png')
    
    # Create and save pie chart without text for Problem Distribution
    plot_pie_chart_no_text(problems_counter, 'data/images/problems_distribution_pie_no_text.png')
    
    print("\nTo view the charts, open the PNG files in your file browser.")

if __name__ == "__main__":
    main()
