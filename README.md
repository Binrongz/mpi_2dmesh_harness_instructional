# CSC 746 CP6 - MPI Sobel Filter

**Student:** [Your Name]  
**Date:** November 2025

---

## Overview

MPI-based distributed Sobel edge detection filter with three domain decomposition strategies (row-slab, column-slab, tiled).

**Extra Credit:** Includes halo cells implementation for correct tile boundary computation.

---

## Files Included

- `mpi_2dmesh.cpp` - Main implementation
- `mpi_2dmesh.hpp` - Header file
- `CMakeLists.txt` - Build configuration
- `README.md` - This file
- `scripts/run_script.sh` - Batch submission script (optional)

---

## Input Data

Input data **NOT included** due to size.

**To obtain:**
```bash
git clone https://github.com/SFSU-Bethel-Instructional/mpi_2dmesh_harness_instructional
```
Use the `data/` directory from the repository, place at `../data/` relative to build directory.

**Required file:** `data/zebra-gray-int8-4x` (7112×5146 pixels)

---

## Build Instructions

### On Perlmutter@NERSC
```bash
module load cpu
export CC=cc
export CXX=CC
export MPICH_GPU_SUPPORT_ENABLED=0

mkdir build && cd build
cmake ..
make
```

### On Other Systems
```bash
mkdir build && cd build
cmake ..
make
```

---

## Running the Code

### Command Line
```bash
./mpi_2dmesh -i <input> -x <width> -y <height> -g <decomp>
```

**Key options:**
- `-g 1` : row-slab decomposition
- `-g 2` : column-slab decomposition
- `-g 3` : tiled decomposition

---

## Reproducing Paper Results

### Interactive Test (Perlmutter)
```bash
# Request node
salloc --nodes 1 --qos interactive --time 00:30:00 \
       --constraint cpu --account=xxxx

# Test single configuration
srun -n 4 ./mpi_2dmesh -i ../data/zebra-gray-int8-4x \
     -x 7112 -y 5146 -g 1
```

### Batch Test All Configurations
```bash
# Run all 24 configurations (8 concurrency × 3 decompositions)
sbatch --account=xxxx ../scripts/run_script.sh ./mpi_2dmesh

# Monitor
sqs
tail -f slurm-<jobid>.out
```

### Extract Results
```bash
# View all timing results
grep "Timing results" -A 3 slurm-<jobid>.out

# Count tests (should be 24)
grep -c "Timing results" slurm-<jobid>.out

# Check for errors
grep -i error slurm-<jobid>.out
```

---

## Expected Output

Each test outputs:
```
Hello world, I'm rank 0 of 4 total ranks running on <node>
...
Timing results from rank 0:
    Scatter time:   XX.XXXX (ms)
    Sobel time:     XX.XXXX (ms)
    Gather time:    XX.XXXX (ms)
```

---

## Performance Metrics Collection

### Runtime Data
- Extract from "Timing results" in slurm output
- Collect for all 24 configurations (8 concurrency × 3 decompositions)

### Message Count Calculation
```
Row-slab:    messages = (nprocs-1) × (image_height/nprocs) × 2
Column-slab: messages = (nprocs-1) × image_height × 2  
Tiled:       messages = (nprocs-1) × (image_height/√nprocs) × 2
```

### Data Transfer Calculation
```
data_MB = (nprocs-1) × tile_width × tile_height × 4 bytes × 2 / (1024²)
```

---

## Troubleshooting

**"Cannot find input file"**
- Ensure `../data/zebra-gray-int8-4x` exists

**"GPU_SUPPORT_ENABLED error"**
```bash
export MPICH_GPU_SUPPORT_ENABLED=0
```

**Poor column-slab performance**
- Expected behavior (non-contiguous memory access)

---

## Implementation Notes

- **MPI communication:** Row-by-row using `MPI_Send`/`MPI_Recv`
- **Sobel kernel:** 3×3 convolution
- **Boundary handling:** Set to 0.0 (basic) or computed with halo cells (extra credit)

---

## Contact

[Your Email]

---

## References

- Code harness: https://github.com/SFSU-Bethel-Instructional/mpi_2dmesh_harness_instructional
- Perlmutter docs: https://docs.nersc.gov/systems/perlmutter/