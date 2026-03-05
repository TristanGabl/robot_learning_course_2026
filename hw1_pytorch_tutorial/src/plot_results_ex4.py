import nbformat
import re
import matplotlib.pyplot as plt
import pandas as pd

def extract_logs_from_notebook(notebook_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    data = []
    # Regex to capture: Run Name, Epoch, Test Acc, and Train Loss
    pattern = re.compile(r"\[(\w+)\] epoch (\d+)/\d+ \| test acc: ([\d.]+)\| train loss: ([\d.]+)")

    for cell in nb.cells:
        if cell.cell_type == 'code':
            for output in cell.get('outputs', []):
                if 'text' in output:
                    text = output['text']
                    matches = pattern.findall(text)
                    for match in matches:
                        data.append({
                            'Run': match[0],
                            'Epoch': int(match[1]),
                            'Test Acc': float(match[2]),
                            'Train Loss': float(match[3])
                        })
    
    return pd.DataFrame(data)

def plot_averaged_results(df):
    # Group by Run and Epoch to calculate mean and standard deviation
    agg_df = df.groupby(['Run', 'Epoch']).agg({
        'Test Acc': ['mean', 'std'],
        'Train Loss': ['mean', 'std']
    }).reset_index()

    runs = agg_df['Run'].unique()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for run in runs:
        # Get the aggregated data for this specific run type
        subset = agg_df[agg_df['Run'] == run].sort_values('Epoch')
        epochs = subset['Epoch']
        
        # Extract mean and standard deviation (fill NaN with 0 for single runs)
        acc_mean = subset[('Test Acc', 'mean')]
        acc_std = subset[('Test Acc', 'std')].fillna(0)
        
        loss_mean = subset[('Train Loss', 'mean')]
        loss_std = subset[('Train Loss', 'std')].fillna(0)

        # Plot Test Accuracy with shaded variance
        ax1.plot(epochs, acc_mean, label=f'{run} (Final Mean: {acc_mean.iloc[-1]:.4f})', marker='o')
        ax1.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)
        
        # Plot Train Loss with shaded variance
        ax2.plot(epochs, loss_mean, label=f'{run}', marker='s')
        ax2.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2)

    # Formatting Accuracy Plot
    ax1.set_title('Epoch vs Average Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Formatting Loss Plot
    ax2.set_title('Epoch vs Average Train Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Print Summary Table
    print("\n--- Final Model Performance (Averaged Across Runs) ---")
    
    # Get the last epoch for each run to print the final stats
    final_epochs = agg_df.groupby('Run')['Epoch'].max().reset_index()
    final_stats = pd.merge(final_epochs, agg_df, on=['Run', 'Epoch'])
    
    # Clean up the multi-index columns for the printout
    summary = pd.DataFrame({
        'Run': final_stats['Run'],
        'Final Epoch': final_stats['Epoch'],
        'Mean Test Acc': final_stats[('Test Acc', 'mean')],
        'Std Test Acc': final_stats[('Test Acc', 'std')].fillna(0),
        'Mean Train Loss': final_stats[('Train Loss', 'mean')],
        'Std Train Loss': final_stats[('Train Loss', 'std')].fillna(0)
    }).set_index('Run')
    
    print(summary)

# Usage
notebook_file = "src/ex4.ipynb" # Adjusted to match your directory structure
df = extract_logs_from_notebook(notebook_file)

if not df.empty:
    plot_averaged_results(df)
else:
    print("No data extracted. Please make sure the output cell text matches the regex format.")