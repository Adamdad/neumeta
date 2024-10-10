import pandas as pd
import re

file_path = "/Users/xingyiyang/Documents/Projects/comptask/Multi-Task-Learning-PyTorch/toy/prune/test_results_32-64_celeba.csv"
def convert_txt(file_path):
    # Raw data as a string
    raw_data = open(file_path).read()# Add the rest of your data here

    # Split the raw data into lines
    lines = raw_data.strip().split('\n')

    # Prepare a list to hold the data
    data = []

    # Go through each line and extract the valuable information
    for line in lines:
        # Use regular expressions to identify and extract the needed parts
        match = re.match(r"Hidden_dim: (\d+), Validation Loss: ([\d.]+), Validation Accuracy: ([\d.]+)%", line)
        # Pruning Ratio,Resulting Channels,Validation Loss,Accuracy
        if match:
            hidden_dim, val_loss, val_acc = match.groups()
            record = {
                'Pruning Ratio': 1-float(hidden_dim)/64.0,
                'Resulting Channels': int(hidden_dim),
                'Validation Loss': float(val_loss),
                'Accuracy': float(val_acc)
            }
            data.append(record)

    # Now, data is a list of dicts, and you can create a DataFrame from it
    df = pd.DataFrame(data)

    # If you need to save this DataFrame as a CSV file, you can do the following:
    df.to_csv('./pruning_results_ResNet20_WINR_cifar100.txt', index=False)

    # If you just want to use the DataFrame for further analysis, you can continue from here.
    
def convert_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    output_file_path = 'toy/prune/pruning_results_VAE_WINR_celeba_vae.txt'
    
    # List to hold the data records
    records = []

    # Iterate over DataFrame rows
    for index, row in df.iterrows():
        # Create a dictionary for each row
        record = {
            'Pruning Ratio': 1 - row['h_dim'] / 64.0,
            'Resulting Channels': int(row['h_dim']),
            'MSE': row['mse'],
            'NLL': row['nll'],
            'MMD': row['mmd']
        }
        records.append(record)
    
    # Create a DataFrame from the list of dictionaries
    new_df = pd.DataFrame(records)
    
    # Save the new DataFrame to a CSV file
    new_df.to_csv(output_file_path, index=False)
    
convert_csv(file_path)

