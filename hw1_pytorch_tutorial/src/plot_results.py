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

def plot_results(df):
    runs = df['Run'].unique()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for run in runs:
        subset = df[df['Run'] == run].sort_values('Epoch')
        
        # Plot Test Accuracy
        ax1.plot(subset['Epoch'], subset['Test Acc'], label=f'{run} (Final: {subset["Test Acc"].iloc[-1]:.4f})', marker='o')
        
        # Plot Train Loss
        ax2.plot(subset['Epoch'], subset['Train Loss'], label=f'{run}', marker='s')

    # Formatting Accuracy Plot
    ax1.set_title('Epoch vs Test Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Formatting Loss Plot
    ax2.set_title('Epoch vs Train Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Print Summary Table
    print("\n--- Final Model Performance ---")
    final_stats = df.groupby('Run').last()[['Test Acc', 'Train Loss']]
    print(final_stats)

# Usage
notebook_file = "src/ex4.ipynb" 
df = extract_logs_from_notebook(notebook_file)
plot_results(df)