# GreedyDogSolver

A Python class for solving DOG (Optimal GPU Distribution) problems using greedy algorithms and iterative improvement methods based on the Iterated Greedy Algorithm meta-heuristic.

## Project Overview
- **Documentation**: A detailed explanation of the problem and algorithm implementation is available in `report.pdf` (in Portuguese)
- **Problem Instances**: Sample problem instances are provided in the `instances` folder
- **Implementation**: Written in Python 3.10.12
- **Dependencies**: All required packages are listed in `requirements.txt`

## Running with Greedy Algorithm (GreedyDog)

Using the `runner_greedydog.py` script:

```bash
python3 runner_greedydog.py <output> <instance> [options]
```

### Required Arguments

- `output`: Path to the output file where the best solution will be saved
- `instance`: Path to the problem instance file

### Options

- `-i, --iterations`: Maximum number of iterations (default: 100000)
- `-t, --temperature`: Initial temperature for the solver (default: 0.3)
- `-s, --seed`: Seed for random number generation (default: None)
- `-e, --enhanced`: Use enhanced initial solution to improve performance
- `-p, --plot`: Plot initial and final solution distribution

### Usage Examples

```bash
# Basic execution
python3 runner_greedydog.py output.csv instance.txt

# Execution with custom parameters
python3 runner_greedydog.py output.csv instance.txt -i 50000 -t 0.5 -s 42 -e -p
```

## Running with Gurobi (Commercial Solver)

Using the `runner_gurobi.py` script:

```bash
python3 runner_gurobi.py <output> <instance> [options]
```

### Required Arguments

- `output`: Path to the output file where the best solution will be saved
- `instance`: Path to the problem instance file

### Options

- `-t, --time`: Time limit for optimization in seconds (default: 100000)

### Usage Examples

```bash
# Basic execution
python3 runner_gurobi.py output.csv instance.txt

# Execution with custom time limit (1 hour)
python3 runner_gurobi.py output.csv instance.txt -t 3600
```

## Main Methods

### Core Features

- `__init__(filename)`: Initializes the solver with problem instance from file
- `solve(...)`: Main solving method with multiple parameters for customization
- `initial_solution()`: Basic greedy allocation method
- `enhanced_initial_solution()`: Enhanced initial allocation with type-based grouping
- `iterated_greedy(...)`: Iterative improvement method using Iterated Greedy Algorithm
- `optimize_gurobi(...)`: Exact solution using Gurobi solver

### Helper Methods

- `mix_noloss()`: Combines GPUs without exceeding VRAM limits
- `mix_loss()`: Merges GPUs when some capacity loss is acceptable
- `avaluate_solution()`: Calculates solution quality based on type distribution

### Analysis and Output

- `save_solution()`: Exports solution to CSV file
- `print_instance_info()`: Displays problem instance details
- `print_gpus_info()`: Shows current GPU allocation information
- `plot_distribution()`: Creates visualizations of current solution