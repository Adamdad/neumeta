import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns


def load_results(folder_path, file_pattern):
    # Construct the full pattern with the folder path
    full_pattern = os.path.join(folder_path, file_pattern)
    
    # Load all files that match the pattern
    files = glob.glob(full_pattern)
    
    # Mapping from filename substrings to formal names
    key_mapping = {
       "Random": "Random Pruning",
        "L1": "L1 Norm Pruning",
        "L2": "L2 Norm Pruning",
        "Taylor": "Taylor-based Pruning",
        "Hessian": "Hessian-based Pruning",
        "NeuMeta": "NeuMeta"
        # Add more mappings as needed
    }

    results = {}
    for filename in files:
        basename = os.path.splitext(os.path.basename(filename))[0]
        
        # Replace the key with its formal name, if it exists in the mapping; otherwise, use the base name
        print(basename.split('_'))
        print(basename.split('_')[3])
        key = key_mapping.get(basename.split('_')[3], basename)     
        
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(filename)
        # Select
        # df = df[(df['Resulting Channels'] >= 4) & df['Resulting Channels'] <= 32]
        df = df[(df['Resulting Channels'] >= 4) & (df['Resulting Channels'] <= 64)]
        # df = df['Pruning Ratio'] > 0.0 & df['Pruning Ratio'] < 1.0
        
        results[key] = df
    
    return results

def plot_results(results):
    # Create a new figure and set the size
    # Set style
    sns.set(style="whitegrid")
    sns.color_palette("hls", 8)
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

    # Create a new figure and set the size
    plt.figure(figsize=(12, 8))

    # Create color palette
    palette = sns.color_palette("tab10", len(results))
     # Define a style cycle for line styles. You can add more if you have many series.
    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # Avoid 'k' (black) because it's used for highlighting
    line_styles = ['-', '--', '-.']
    markers = ['o', 'v', '^', '<', '>', 's']
    title_fontsize = 16  # for the title text
    label_fontsize = 40  # for the axis labels (i.e., 'Pruning Ratio (%)' and 'Accuracy (%)')
    ticks_fontsize = 20  # for the axis ticks (i.e., the numbers along the axes)
    legend_fontsize = 23 # for the legend

    # print(results.keys())
    preferred_order = ['NeuMeta', 'Random Pruning', 'L1 Norm Pruning', 'L2 Norm Pruning', 'Taylor-based Pruning', 'Hessian-based Pruning']
    sorted_results = {key: results[key] for key in preferred_order if key in results}
    # Plot each set of results
    for (idx, (key, df)) in enumerate(sorted_results.items()):
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        # Plot with different line styles and markers for each set of results.
        if key == 'NeuMeta':
            plt.plot(df['Pruning Ratio']* 100, df['Accuracy'], 
                    label=f'{key}', 
                    color='#E14A1F',
                    linewidth=2,
                    linestyle=line_style,  # Different line styles
                    marker='D',  # Different markers
                    markerfacecolor= 'gold',
                    markersize=10,   # You can change this value as you see fit
                    markevery=5)    # This controls at which data points the marker should appear. This is useful if there are many data points.

                     
        else:
            plt.plot(df['Pruning Ratio'] * 100, df['Accuracy'], # - 5 
                    label=f'{key}', 
                    linestyle=line_style,  # Different line styles
                    marker=marker,  # Different markers
                    markersize=7,   # You can change this value as you see fit
                    markevery=5)    # This controls at which data points the marker should appear. This is useful if there are many data points.

    # plt.title('Pruning Ratio vs Accuracy Comparison')
    plt.xlabel('Compress Ratio (%)',fontsize=label_fontsize)
    plt.ylabel('Accuracy (%)',fontsize=label_fontsize)  # Add units if applicable
    plt.legend(loc='lower left',fontsize=legend_fontsize)  # You can change the location of the legend
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('imagenet_resnet18_ratio_vs_accuracy_comparison.pdf', dpi=300)


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_results_vae_bar(results):
    # Set style
    sns.set(style="whitegrid")
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})

    # Create a new figure and set the size
    plt.figure(figsize=(12, 9))

    # Fonts sizes
    label_fontsize = 40  # for the axis labels (i.e., 'Pruning Ratio (%)' and 'Accuracy (%)')
    ticks_fontsize = 20  # for the axis ticks (i.e., the numbers along the axes)
    legend_fontsize = 26 # for the legend

    # Merge all results into one DataFrame
    all_results = pd.DataFrame()
    for method, df in results.items():
        df_temp = df.copy()
        df_temp['Method'] = method
        all_results = pd.concat([all_results, df_temp])

    # Filter for specific values of 'Resulting Channels'
    filter_channels = [64, 48, 38, 32]
    filtered_results = all_results[all_results['Resulting Channels'].isin(filter_channels)]
    filtered_results['Pruning Ratio'] = (filtered_results['Pruning Ratio']*100).astype(int)
    preferred_order = ['NeuMeta', 'Random Pruning', 'L1 Norm Pruning', 'L2 Norm Pruning', 'Taylor-based Pruning', 'Hessian-based Pruning']
    # Sort so 'WINR' is last in the 'Method' column
    filtered_results['Method'] = pd.Categorical(filtered_results['Method'], 
                                                categories=preferred_order,
                                                ordered=True)

    # Use seaborn's barplot with a sequential palette
    palette = sns.color_palette("pastel", len(filtered_results['Method'].unique()))
    palette = ['#E14A1F'] + palette
    print(palette)
    # palette = sns.color_palette("hls", 8)
    # palette = sns.color_palette("GnBu", len(filtered_results['Method'].unique()))
    ax = sns.barplot(x='Pruning Ratio', y='NLL', hue='Method', data=filtered_results, palette=palette)

    # plt.ylim(100, 350)
    plt.ylim(200, 1150)
    # Set the labels and legend
    plt.xlabel('Compress Ratio (%)', fontsize=label_fontsize)
    plt.ylabel('Negative Log-Likelihood', fontsize=label_fontsize)  # Add units if applicable
    plt.legend(loc='upper left', fontsize=legend_fontsize,framealpha=0.5)  # You can change the location of the legend
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig('celeba_vae_ratio_vs_nll_comparison.pdf', dpi=300)


    
def main():
    # Specify the folder containing the results files
    folder_path = '/Users/xingyiyang/Documents/Projects/comptask/Multi-Task-Learning-PyTorch/toy/prune/'  # e.g., 'results/' or '/home/user/experiments/results/'
    
    # Load results from text files in the specified folder. The '*' is a wildcard that matches any character.
    file_pattern = '*_results_*.txt'  # This will match files like "method1_results.txt", "method2_results.txt", etc.
    
    results = load_results(folder_path, file_pattern)

    # plot_results(results)
    # Plot the results
    plot_results_vae_bar(results)

if __name__ == "__main__":
    main()
