from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import csv
from typing import List, Dict
import random
import time
import math

class GreedyDogSolver:
    def __init__(self, filename:str) -> None:
        self.name = filename.split('/')[-1].rsplit('.',1)[0]
        self.gpu_n = 0
        self.gpu_vram = 0
        self.prn_types_n = 0
        self.prn_n = 0
        self.prns = []
        self.gpus = []

        try:
            with open(filename, 'r') as file:
                self.gpu_n = int(file.readline())
                self.gpu_vram = int(file.readline())
                self.prn_types_n = int(file.readline())
                self.prn_n = int(file.readline())
                self.gpus = [{'prns': [], 'occupied_vram': 0} for _ in range(self.gpu_n)]

                for _ in range(self.prn_n):
                    row = file.readline()
                    parsed_row = row.strip().split('\t')
                    type, vram = int(parsed_row[0]), int(parsed_row[1])
                    self.prns.append({"type": type, "vram": vram})

                self.prns.sort(key=lambda x: x['vram'], reverse=True)
                self.prns.sort(key=lambda x: x['type'])
        except Exception as e:
            self.name = None
            raise RuntimeError(f"Error: Unable to read the file \"{filename}\". Details: {str(e)}")

    def solve(self, output_file:str="output.txt", max_iterations:int=100000, temperature:float=1.0, seed:int=None, enhanced:bool=False, plot:bool=False) -> None:
        if not hasattr(self, 'name') or self.name is None:
            raise ValueError("Error: The instance file was not loaded. Please ensure the instance file is read before proceeding.")

        self.print_instance_info()

        print("\nRunning initial solution...")
        if enhanced:
            self.enhanced_initial_solution()
        else:
            self.initial_solution()

        self.initial_solution_value = self.avaluate_solution(self.gpus)
        self.print_gpus_info()

        if plot:
            self.plot_distribution()

        print("\nRunning heuristic...")
        start_time = time.time()
        self.iterated_greedy(max_iterations=max_iterations, temperature=temperature, seed=seed)
        end_time = time.time()
        print(f"\nSeed: {self.seed}")
        self.print_gpus_info()

        if plot:
            self.plot_distribution()

        self.save_solution(filename=output_file)

        self.execution_time = end_time - start_time

        print(f"\nExecution time: {self.execution_time:.2f} seconds\n")


    def initial_solution(self) -> None:
        fits_on_gpu = lambda gpu_idx, prn: self.gpus[gpu_idx]['occupied_vram'] + prn['vram'] <= self.gpu_vram

        for prn_index, prn in enumerate(self.prns):
            gpu_index = 0
            # Find the first GPU where the PRN fits
            while not fits_on_gpu(gpu_index, prn):
                gpu_index += 1
                # Add a new GPU if necessary
                if gpu_index >= len(self.gpus):
                    self.gpus.append({'prns': [], 'occupied_vram': 0})

            # Assign the PRN to the GPU and update VRAM usage
            self.gpus[gpu_index]['prns'].append(prn_index)
            self.gpus[gpu_index]['occupied_vram'] += prn['vram']


    def initial_solution_steroids(self) -> None:
        gpu_index = 0
        current_type = self.prns[0]['type']
        start_index = 0

        # Lambda function to check if a PRN fits on a GPU
        fits_on_gpu = lambda gpu_idx, prn: self.gpus[gpu_idx]['occupied_vram'] + prn['vram'] <= self.gpu_vram

        for prn_index, prn in enumerate(self.prns):
            # Update GPU index if the PRN type changes
            if current_type != prn['type']:
                # Find empty GPU
                while len(self.gpus[gpu_index]['prns']) > 0:
                    gpu_index += 1
                    if gpu_index >= len(self.gpus):
                        self.gpus.append({'prns': [], 'occupied_vram': 0})
                current_type = prn['type']
                start_index = gpu_index

            gpu_index = start_index

            # Find the first GPU where the PRN fits
            while not fits_on_gpu(gpu_index, prn):
                gpu_index += 1
                # Add a new GPU if necessary
                if gpu_index >= len(self.gpus):
                    self.gpus.append({'prns': [], 'occupied_vram': 0})

            # Assign the PRN to the GPU and update VRAM usage
            self.gpus[gpu_index]['prns'].append(prn_index)
            self.gpus[gpu_index]['occupied_vram'] += prn['vram']
        
        if(len(self.gpus) > self.gpu_n):
            self.mix_noloss()
            self.mix_loss()
    
    def mix_noloss(self) -> None:
        # Sort GPUs by occupied VRAM
        self.gpus.sort(key=lambda gpu: gpu['occupied_vram'], reverse=True)


        can_merge_gpus = lambda gpu1, gpu2: gpu1['occupied_vram'] + gpu2['occupied_vram'] <= self.gpu_vram
        new_gpus = [{'prns': [], 'occupied_vram': 0} for _ in range(self.gpu_n)]

        for gpu in self.gpus:

            new_gpu_idx = 0
            while not can_merge_gpus(gpu, new_gpus[new_gpu_idx]):
                new_gpu_idx += 1
                # Add a new GPU if necessary
                if new_gpu_idx >= len(new_gpus):
                    new_gpus.append({'prns': [], 'occupied_vram': 0})
                
            
            new_gpus[new_gpu_idx]['prns'] += gpu['prns']
            new_gpus[new_gpu_idx]['occupied_vram'] += gpu['occupied_vram']
        self.gpus = new_gpus

    
    def mix_loss(self) -> None:
        def destroy() -> List[Dict[List[int], int]]:
            self.gpus.sort(key=lambda gpu: gpu['occupied_vram'])
            prns = self.gpus[0]['prns'] + self.gpus[1]['prns']
            prns.sort(key=lambda prn_idx: self.prns[prn_idx]['vram'], reverse=True)
            self.gpus.pop(0)
            self.gpus.pop(0)
            return prns
        
        def construct(prns):
            new_gpus = [{'prns': [], 'occupied_vram': 0}]
            for prn_idx in prns:
                gpu_idx = 0
                if self.prns[prn_idx]['vram'] + new_gpus[0]['occupied_vram'] >= self.gpu_vram:
                    gpu_idx += 1
                    if len(new_gpus) >= 1:
                        new_gpus.append({'prns': [], 'occupied_vram': 0})
                new_gpus[gpu_idx]['prns'].append(prn_idx)
                new_gpus[gpu_idx]['occupied_vram'] += self.prns[prn_idx]['vram']
            self.gpus += new_gpus

        while len(self.gpus) > self.gpu_n:
            prns = destroy()
            construct(prns)

    def save_solution(self, filename: str) -> None:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['PRN Index', 'PRN VRAM', 'PRN Type', 'GPU Index'])

            for i in range(len(self.gpus)):
                for j in self.gpus[i]['prns']:
                    writer.writerow([j, self.prns[j]['vram'], self.prns[j]['type'], i])

    def avaluate_solution(self, gpus) -> int:
        def gpu_type_distribution(gpu):
            types = []
            for prn_index in gpu['prns']:
                if self.prns[prn_index]['type'] not in types:
                    types.append(self.prns[prn_index]['type'])
            return len(types)

        value = 0
        for gpu in gpus:
            value += gpu_type_distribution(gpu)
        return value
    
    def iterated_greedy(self, max_iterations:int=100000, temperature:float=1.0, seed:int=None) -> None:
        self.seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        random.seed(self.seed)

        def destroy(self, gpus) -> List[int]:
            # Make a copy of gpus list to avoid modifying the original during random.choice
            available_gpus = gpus.copy()
            
            # Select first GPU
            gpu1 = random.choice(available_gpus)
            available_gpus.remove(gpu1)
            
            # Select second GPU from remaining GPUs
            gpu2 = random.choice(available_gpus)
            available_gpus.remove(gpu2)
            
            # Select third GPU from remaining GPUs
            gpu3 = random.choice(available_gpus)
            
            # print(f'destroy: \n - {gpu1}\n - {gpu2}\n - {gpu3}\n')
            
            current_solution = self.avaluate_solution([gpu1, gpu2, gpu3])
            prns = gpu1['prns'] + gpu2['prns'] + gpu3['prns']
            
            # Remove selected GPUs from original list
            gpus.remove(gpu1)
            gpus.remove(gpu2)
            gpus.remove(gpu3)
            
            return prns, current_solution

        def construct(self, prns, gpus):
            prns.sort(key=lambda prn_idx: self.prns[prn_idx]['type'])
            
            new_gpus = [{'prns': [], 'occupied_vram': 0}]
            
            # Define fits_on_gpu lambda outside the loop
            fits_on_gpu = lambda gpu_idx, prn_idx: new_gpus[gpu_idx]['occupied_vram'] + self.prns[prn_idx]['vram'] <= self.gpu_vram
            
            # Iterate through PRN indices
            for prn_idx in prns:  # Remove enumerate since prns appears to be a list of indices
                gpu_index = 0
                
                # Find the first GPU where the PRN fits
                while not fits_on_gpu(gpu_index, prn_idx):
                    gpu_index += 1
                    
                    # Add a new GPU if necessary
                    if gpu_index >= len(new_gpus):
                        new_gpus.append({'prns': [], 'occupied_vram': 0})
                
                # Assign the PRN to the GPU and update VRAM usage
                new_gpus[gpu_index]['prns'].append(prn_idx)
                new_gpus[gpu_index]['occupied_vram'] += self.prns[prn_idx]['vram']
            
            new_value = self.avaluate_solution(new_gpus)
            gpus += new_gpus
            return new_value, gpus
        
        def accept_solution(delta, temperature):
            probability = math.exp(-delta / temperature)
            return random.random() < min(1, probability)
        
        current_solution_global = self.avaluate_solution(self.gpus)
        best_solution = current_solution_global
        best_gpus = self.gpus

        # Process each type separately
        for _ in range(max_iterations):
            gpus = self.gpus.copy()
            prns, current_solution = destroy(self, gpus)
            new_solution, gpus = construct(self, prns, gpus)
            
            delta = new_solution - current_solution
            
            if len(gpus) <= self.gpu_n and accept_solution(delta, temperature):
                current_solution_global = current_solution_global - current_solution + new_solution
                self.gpus = gpus 
                if(current_solution_global < best_solution):
                    best_solution = current_solution_global
                    best_gpus = gpus

        self.gpus = best_gpus
        self.best_solution_value = best_solution
    
    def optimize_gurobi(self, time_limit:int=1800) -> None:
        n = self.gpu_n
        m = self.prn_n
        V = self.gpu_vram
        v = []
        t = []
        for prn in self.prns:
            v.append(int(prn["vram"]))
            t.append(int(prn["type"]))
        types = range(self.prn_types_n)

        model = Model(self.name)

        x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
        y = model.addVars(n, len(types), vtype=GRB.BINARY, name="y")

        # Objective Function: Total type distribution
        model.setObjective(y.sum(), GRB.MINIMIZE)

        # Constraint (1): Limits VRAM capacity
        for i in range(n):
            model.addConstr(sum(x[i, j] * v[j] for j in range(m)) <= V, name=f"VRAM_{i}")

        # Constraint (2): Each PRN have to be processed by one GPU
        for j in range(m):
            model.addConstr(sum(x[i, j] for i in range(n)) == 1, name=f"AssignPRN_{j}")

        # Constraint (3): x, y connection
        for i in range(n):
            for j in range(m):
                prn_type_index = types.index(t[j])
                model.addConstr(x[i, j] <= y[i, prn_type_index], name=f"Link_x_y_{i}_{j}")

        model.setParam('TimeLimit', time_limit)
        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print("\nOptimal Solution Found:")
        else:
            print("\nOptimal solution not found.")

        print(f"Total type distribution: {model.ObjVal}")

        # Extract the solution and write to CSV
        with open(f'gurobi_solutions/solution_{self.name}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['PRN Index', 'PRN VRAM', 'PRN Type', 'GPU Index'])

            for j in range(m):
                for i in range(n):
                    if x[i, j].X > 0.5:
                        writer.writerow([j, v[j], t[j], i])
                        break
    
    def print_instance_info(self) -> None:
        print("\n=============================")
        print("Dog Instance Info")
        print(self.name)
        print("=============================")
        print(f"Number of GPU's: {self.gpu_n}")
        print(f"GPU's VRAM: {self.gpu_vram}")
        print(f"Number of PRN's: {self.prn_n}")
        print(f"Number of PRN's types: {self.prn_types_n}")

        max_vram_prn = max(self.prns, key=lambda x: x['vram'])
        min_vram_prn = min(self.prns, key=lambda x: x['vram'])
        total_prn_vram = sum(prn['vram'] for prn in self.prns)
        total_gpu_vram = self.gpu_n * self.gpu_vram

        print(f"Max PRN VRAM: {max_vram_prn['vram']} (Type: {max_vram_prn['type']})")
        print(f"Min PRN VRAM: {min_vram_prn['vram']} (Type: {min_vram_prn['type']})")
        print(f"Total PRN's VRAM: {total_prn_vram}")
        print(f"Total GPU's VRAM: {total_gpu_vram}\n")

    def print_prns(self) -> None:
        print("\n=============================")
        print("PRN's")
        print(self.name)
        print("=============================")
        for i, prn in enumerate(self.prns):
            print(f"[{i}] : {prn}")

    def print_gpus_info(self) -> None:
        print("\n=============================")
        print("Dog GPU's Info")
        print(self.name)
        print("=============================")

        def gpu_type_distribution(gpu):
            types = []
            for prn_index in gpu['prns']:
                if self.prns[prn_index]['type'] not in types:
                    types.append(self.prns[prn_index]['type'])
            return len(types)


        print(f"Sum of type distibution for all GPU\'s: {self.avaluate_solution(self.gpus)}\n")
        empty_gpus = 0
        for gpu in self.gpus:
            if len(gpu['prns']) == 0:
                empty_gpus += 1

        extra = len(self.gpus)-self.gpu_n

        print(f"Number of extra GPU's: {extra if empty_gpus >= 0 else 0}")
        print(f"Number of empty GPU's: {empty_gpus}")

        max_occupied_vram= max(self.gpus, key=lambda x: x['occupied_vram'])
        min_occupied_vram = min(self.gpus, key=lambda x: x['occupied_vram'])

        print(f"Max GPU occupied VRAM: {max_occupied_vram['occupied_vram']} (Type distribution: {gpu_type_distribution(max_occupied_vram)})")
        print(f"Min GPU occupied VRAM: {min_occupied_vram['occupied_vram']} (Type distribution: {gpu_type_distribution(min_occupied_vram)})\n")

    def plot_distribution(self) -> None:
            """
            Plot the type distribution and occupied VRAM for each GPU.
            Requires matplotlib to be installed.
            """

            # Calculate type distribution for each GPU
            type_distributions = []
            occupied_vrams = []
            number_prns = []

            for gpu in self.gpus:
                # Get unique types for this GPU
                types = set()
                for prn_index in gpu['prns']:
                    types.add(self.prns[prn_index]['type'])
                type_distributions.append(len(types))
                occupied_vrams.append(gpu['occupied_vram'])
                number_prns.append(len(gpu['prns']))

            # Create figure with two subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

            # GPU indices for x-axis
            gpu_indices = range(len(self.gpus))

            # Plot type distribution
            ax1.bar(gpu_indices, type_distributions, color='skyblue')
            ax1.set_title('Type Distribution per GPU')
            ax1.set_xlabel('GPU Index')
            ax1.set_ylabel('Number of Different Types')
            ax1.grid(True, alpha=0.3)

            # Plot occupied VRAM
            ax2.bar(gpu_indices, occupied_vrams, color='lightgreen')
            ax2.axhline(y=self.gpu_vram, color='red', linestyle='--', label='VRAM Limit')
            ax2.axvline(x=self.gpu_n-1, color='black', linestyle='--', label='Last Real GPU')
            ax2.set_title('Occupied VRAM per GPU')
            ax2.set_xlabel('GPU Index')
            ax2.set_ylabel('VRAM')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            # Plot number of PRN's
            ax3.bar(gpu_indices, number_prns, color='tomato')
            ax3.set_title('Number of PRN\'S per GPU')
            ax3.set_xlabel('GPU Index')
            ax3.set_ylabel('Number of PRN\'s')
            ax3.grid(True, alpha=0.3)

            # Add overall title
            plt.suptitle(f'GPU Distribution Analysis - {self.name}')

            # Adjust layout and display
            plt.tight_layout()
            plt.show()