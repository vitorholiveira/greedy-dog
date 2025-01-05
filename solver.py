from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import csv
from typing import List, Dict

class GreedyDogSolver:
    def __init__(self, filename: str) -> None:
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

                for i in range(self.prn_n):
                    row = file.readline()
                    parsed_row = row.strip().split('\t')
                    type, vram = int(parsed_row[0]), int(parsed_row[1])
                    self.prns.append({"type": type, "vram": vram})

                self.prns.sort(key=lambda x: x['vram'], reverse=True)
                self.prns.sort(key=lambda x: x['type'])
        except:
            print(f"\nCan't read \"{filename}\"\n")

    def solve(self):
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

        self.mix_noloss()

        self.mix_loss()

        self.print_gpus_info()

        self.plot_distribution()
    
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
            print(f'destroy: \n - {self.gpus[0]}\n - {self.gpus[1]}\n')
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
                    gpu_idx = 1
                    if len(new_gpus) == 1:
                        new_gpus.append({'prns': [], 'occupied_vram': 0})
                new_gpus[gpu_idx]['prns'].append(prn_idx)
                new_gpus[gpu_idx]['occupied_vram'] += self.prns[prn_idx]['vram']
            self.gpus += new_gpus
            print(f'construct: \n - {new_gpus[0]}\n - {new_gpus[1] if len(new_gpus) > 1 else None}\n')
        
        self.print_prns()

        while len(self.gpus) > self.gpu_n:
            print(f'num gpus: {len(self.gpus)}\n')
            prns = destroy()
            construct(prns)

    def optimize_gurobi(self, time_limit=1800) -> None:
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

        def type_distribution_forall_gpus():
            sum = 0
            for gpu in self.gpus:
                sum += gpu_type_distribution(gpu)
            return sum


        print(f"Sum of type distibution for all GPU\'s: {type_distribution_forall_gpus()}\n")
        empty_gpus = 0
        for gpu in self.gpus:
            if len(gpu['prns']) == 0:
                empty_gpus += 1

        print(f"Number of extra GPU's: {len(self.gpus)-self.gpu_n}")
        print(f"Number of empty GPU's: {empty_gpus}")

        max_occupied_vram= max(self.gpus, key=lambda x: x['occupied_vram'])
        min_occupied_vram = min(self.gpus, key=lambda x: x['occupied_vram'])
        total_occupied_vram = sum(gpu['occupied_vram'] for gpu in self.gpus)
        total_gpu_vram = len(self.gpus) * self.gpu_vram

        print(f"Max GPU occupied VRAM: {max_occupied_vram['occupied_vram']} (Type distribution: {gpu_type_distribution(max_occupied_vram)})")
        print(f"Min GPU occupied VRAM: {min_occupied_vram['occupied_vram']} (Type distribution: {gpu_type_distribution(min_occupied_vram)})")
        print(f"Total GPU's occupied VRAM: {total_occupied_vram}")
        print(f"Total GPU's VRAM: {total_gpu_vram}\n")


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