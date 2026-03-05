import nbformat
import re
import matplotlib.pyplot as plt
import pandas as pd

def extract_logs_from_notebook(notebook_path):
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    data = []
    # New regex to capture Epoch and Accuracy
    pattern = re.compile(r"Epoch (\d+), Accuracy: ([\d.]+), Train Loss: ([\d.]+)")
    
    run_counter = 1

    for cell in nb.cells:
        if cell.cell_type == 'code':
            has_matches = False
            code_source = cell.source.replace('\n', ' ')
            
            for output in cell.get('outputs', []):
                if 'text' in output:
                    text = output['text']
                    matches = pattern.findall(text)
                    
                    if matches:
                        # Attempt to extract model config from the code cell to use as the run name
                        model_match = re.search(r'MNIST_classifier\((.*?)\)', code_source)
                        if model_match:
                            run_name = f"Model({model_match.group(1).strip()})"
                        else:
                            run_name = f"Run {run_counter}"
                            
                        # Append each epoch's data
                        for match in matches:
                            data.append({
                                'Run': run_name,
                                'Epoch': int(match[0]),
                                'Accuracy': float(match[1]),
                                'Train Loss': float(match[2])

                            })
                        has_matches = True
                        
            # Increment counter for the next run if we found data in this cell
            if has_matches:
                run_counter += 1
    
    return pd.DataFrame(data)

def plot_results(df):
    if df.empty:
        print("No training logs found in the notebook.")
        return

    runs = df['Run'].unique()
    
    # Create two plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for run in runs:
        subset = df[df['Run'] == run].sort_values('Epoch')
        
        # Plot Accuracy
        ax1.plot(subset['Epoch'], subset['Accuracy'], 
                 label=f'{run} (Final: {subset["Accuracy"].iloc[-1]:.4f})', 
                 marker='o')
        
        # Plot Train Loss
        ax2.plot(subset['Epoch'], subset['Train Loss'],
                 label=f'{run} (Final: {subset["Train Loss"].iloc[-1]:.4f})',
                 marker='o')

    # Formatting Accuracy Plot
    ax1.set_title('Epoch vs Accuracy per Run')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Formatting Train Loss Plot
    ax2.set_title('Epoch vs Train Loss per Run')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Train Loss')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    # Print Summary Table
    print("\n--- Final Model Performance ---")
    final_stats = df.groupby('Run').last()[['Accuracy', 'Train Loss']]
    print(final_stats)

# Usage
notebook_file = "src/ex3.ipynb" 
df = extract_logs_from_notebook(notebook_file)
plot_results(df)