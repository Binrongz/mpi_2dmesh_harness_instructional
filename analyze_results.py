#!/usr/bin/env python3
import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_slurm_output(filename):
    """Parse slurm output file and extract timing results"""
    results = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find all test runs with timing
    # Pattern matches: srun command followed by timing results
    blocks = content.split('srun -n ')
    
    for block in blocks[1:]:  # Skip first empty block
        # Extract nprocs and decomp
        nprocs_match = re.search(r'^(\d+)', block)
        decomp_match = re.search(r'-g (\d+)', block)
        
        # Extract timing
        scatter_match = re.search(r'Scatter time:\s+([\d.]+)', block)
        sobel_match = re.search(r'Sobel time:\s+([\d.]+)', block)
        gather_match = re.search(r'Gather time:\s+([\d.]+)', block)
        
        if all([nprocs_match, decomp_match, scatter_match, sobel_match, gather_match]):
            results.append({
                'Nprocs': int(nprocs_match.group(1)),
                'Decomp': int(decomp_match.group(1)),
                'Scatter_ms': float(scatter_match.group(1)),
                'Sobel_ms': float(sobel_match.group(1)),
                'Gather_ms': float(gather_match.group(1))
            })
    
    return pd.DataFrame(results)

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    print(f"\nTotal test runs: {len(df)}")
    print(f"Expected: 24 (8 concurrency levels × 3 decompositions)")
    
    if len(df) == 24:
        print("✓ All tests completed!")
    else:
        print("✗ WARNING: Missing some test runs!")
    
    print("\nConcurrency levels tested:", sorted(df['Nprocs'].unique()))
    print("Decomposition strategies tested:", sorted(df['Decomp'].unique()))
    
    print("\n" + "-"*60)
    print("Timing Statistics (milliseconds)")
    print("-"*60)
    print("\nScatter Time:")
    print(df['Scatter_ms'].describe())
    
    print("\nSobel Time:")
    print(df['Sobel_ms'].describe())
    
    print("\nGather Time:")
    print(df['Gather_ms'].describe())
    
    # Check for anomalies
    print("\n" + "-"*60)
    print("Data Quality Check")
    print("-"*60)
    
    issues = []
    if (df['Scatter_ms'] <= 0).any():
        issues.append("✗ Found zero or negative Scatter times")
    if (df['Sobel_ms'] <= 0).any():
        issues.append("✗ Found zero or negative Sobel times")
    if (df['Gather_ms'] <= 0).any():
        issues.append("✗ Found zero or negative Gather times")
    
    if not issues:
        print("✓ All timing values are positive")
    else:
        for issue in issues:
            print(issue)
    
    print("\n")

def show_data_by_decomp(df):
    """Show data grouped by decomposition strategy"""
    decomp_names = {1: 'Row-slab', 2: 'Column-slab', 3: 'Tiled'}
    
    print("\n" + "="*60)
    print("DETAILED RESULTS BY DECOMPOSITION")
    print("="*60)
    
    for decomp in sorted(df['Decomp'].unique()):
        print(f"\n{decomp_names[decomp]} (decomp={decomp}):")
        print("-"*60)
        subset = df[df['Decomp'] == decomp].sort_values('Nprocs')
        print(subset.to_string(index=False))

def calculate_speedup(df):
    """Calculate speedup relative to baseline (nprocs=4)"""
    print("\n" + "="*60)
    print("SPEEDUP ANALYSIS (Baseline: 4 processes)")
    print("="*60)
    
    for decomp in sorted(df['Decomp'].unique()):
        print(f"\nDecomposition {decomp}:")
        subset = df[df['Decomp'] == decomp].sort_values('Nprocs')
        
        # Get baseline (nprocs=4)
        baseline = subset[subset['Nprocs'] == 4]
        if baseline.empty:
            print("  WARNING: No baseline data (nprocs=4)")
            continue
        
        baseline_scatter = baseline['Scatter_ms'].iloc[0]
        baseline_sobel = baseline['Sobel_ms'].iloc[0]
        baseline_gather = baseline['Gather_ms'].iloc[0]
        
        print(f"  {'Nprocs':<10} {'Scatter':<12} {'Sobel':<12} {'Gather':<12}")
        print(f"  {'-'*46}")
        
        for _, row in subset.iterrows():
            nprocs = row['Nprocs']
            scatter_speedup = baseline_scatter / row['Scatter_ms']
            sobel_speedup = baseline_sobel / row['Sobel_ms']
            gather_speedup = baseline_gather / row['Gather_ms']
            
            print(f"  {nprocs:<10} {scatter_speedup:<12.3f} {sobel_speedup:<12.3f} {gather_speedup:<12.3f}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py slurm-XXXXX.out")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Parse data
    print(f"Parsing {input_file}...")
    df = parse_slurm_output(input_file)
    
    if df.empty:
        print("ERROR: No timing data found in file!")
        sys.exit(1)
    
    # Save to CSV
    output_csv = 'results.csv'
    df.to_csv(output_csv, index=False)
    print(f"✓ Data saved to {output_csv}")
    
    # Print analysis
    print_summary(df)
    show_data_by_decomp(df)
    calculate_speedup(df)
    
    print("\n" + "="*60)
    print("Next steps:")
    print("  1. Review the speedup analysis above")
    print("  2. Check results.csv for detailed data")
    print("  3. Create graphs using the CSV data")
    print("="*60)