#!/usr/bin/env python3
import pandas as pd
import numpy as np

def calculate_messages_and_data(nprocs, decomp, width=7112, height=5146):
    """
    Calculate number of messages and data transferred
    
    Returns: (num_messages, data_mb)
    """
    
    if decomp == 1:  # Row-slab
        # Each rank gets full width, divided height
        tile_width = width
        tile_height = height // nprocs
        
        # Number of tiles not owned by rank 0
        num_tiles = nprocs - 1
        
        # Messages per tile = tile height (one message per row)
        messages_per_tile = tile_height
        
        # Total messages (scatter + gather)
        total_messages = num_tiles * messages_per_tile * 2
        
        # Data transferred (in MB)
        data_per_tile = tile_width * tile_height * 4  # 4 bytes per float
        total_data_mb = (num_tiles * data_per_tile * 2) / (1024 * 1024)
        
    elif decomp == 2:  # Column-slab
        # Each rank gets divided width, full height
        tile_width = width // nprocs
        tile_height = height
        
        num_tiles = nprocs - 1
        
        # Messages per tile = tile height (one message per row)
        messages_per_tile = tile_height
        
        total_messages = num_tiles * messages_per_tile * 2
        
        data_per_tile = tile_width * tile_height * 4
        total_data_mb = (num_tiles * data_per_tile * 2) / (1024 * 1024)
        
    else:  # Tiled (decomp == 3)
        # 2D decomposition: sqrt(nprocs) × sqrt(nprocs)
        sqrt_n = int(np.sqrt(nprocs))
        
        tile_width = width // sqrt_n
        tile_height = height // sqrt_n
        
        num_tiles = nprocs - 1
        
        messages_per_tile = tile_height
        
        total_messages = num_tiles * messages_per_tile * 2
        
        data_per_tile = tile_width * tile_height * 4
        total_data_mb = (num_tiles * data_per_tile * 2) / (1024 * 1024)
    
    return int(total_messages), round(total_data_mb, 2)

# Read timing results
df = pd.read_csv('results.csv')

# Calculate for each configuration
results = []
for _, row in df.iterrows():
    nprocs = row['Nprocs']
    decomp = row['Decomp']
    
    num_msgs, data_mb = calculate_messages_and_data(nprocs, decomp)
    
    results.append({
        'Nprocs': nprocs,
        'Decomp': decomp,
        'Num_Messages': num_msgs,
        'Data_MB': data_mb,
        'Scatter_ms': row['Scatter_ms'],
        'Sobel_ms': row['Sobel_ms'],
        'Gather_ms': row['Gather_ms']
    })

# Create DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('results_complete.csv', index=False)
print("✓ Saved complete results to results_complete.csv")

# Print summary table
print("\n" + "="*80)
print("MESSAGE COUNTS AND DATA MOVEMENT SUMMARY")
print("="*80)

decomp_names = {1: 'Row-slab', 2: 'Column-slab', 3: 'Tiled'}

for decomp in [1, 2, 3]:
    print(f"\n{decomp_names[decomp]}:")
    print("-" * 80)
    subset = results_df[results_df['Decomp'] == decomp].sort_values('Nprocs')
    print(f"{'Nprocs':<10} {'Messages':<15} {'Data (MB)':<15} {'Scatter (ms)':<15} {'Gather (ms)':<15}")
    print("-" * 80)
    for _, row in subset.iterrows():
        print(f"{row['Nprocs']:<10} {row['Num_Messages']:<15,} {row['Data_MB']:<15.2f} "
              f"{row['Scatter_ms']:<15.2f} {row['Gather_ms']:<15.2f}")

# Print comparison
print("\n" + "="*80)
print("COMPARISON AT 81 PROCESSES")
print("="*80)
final = results_df[results_df['Nprocs'] == 81]
print(f"{'Strategy':<15} {'Messages':<15} {'Data (MB)':<15} {'Gather Time (ms)':<20}")
print("-" * 80)
for _, row in final.iterrows():
    print(f"{decomp_names[row['Decomp']]:<15} {row['Num_Messages']:<15,} "
          f"{row['Data_MB']:<15.2f} {row['Gather_ms']:<20.2f}")

print("\n")