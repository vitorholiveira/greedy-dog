from gurobipy import Model, GRB

class GreedyDogSolver:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.gpu_n = None
        self.gpu_vram = None
        self.prn_types_n = None
        self.prn_n = None
        self.prns = []
        self.gpus = []
        
        try:
            with open(filename, 'r') as file:
                self.gpu_n = int(file.readline())
                self.gpu_vram = int(file.readline())
                self.prn_types_n = int(file.readline())
                self.prn_n = int(file.readline())
                
                for i in range(self.prn_n):
                    row = file.readline()
                    parsed_row = row.strip().split('\t')
                    type, vram = int(parsed_row[0]), int(parsed_row[1])
                    self.prns.append({"type": type, "vram": vram})
                
                # Ver se esse sort bombou
                self.prns.sort(key=lambda x: x['vram'])
                self.prns.sort(key=lambda x: x['type'])
        except:
            print(f"\ncan't read \"{filename}\"\n")

    def print_info(self) -> None:
        print("\n=============================")
        print("Dog Instance Info")
        print(self.filename)
        print("=============================")
        print(f"Number of GPU's: {self.gpu_n}")
        print(f"GPU's VRAM: {self.gpu_vram}")
        print(f"Number of PRN's: {self.prn_n}")
        print(f"Number of PRN's types: {self.prn_types_n}")
        print(f"Max PRN VRAM: {max(max(prn_vram) for prn_vram in self.prn.values())}")
        print(f"Min PRN VRAM: {min(min(prn_vram) for prn_vram in self.prn.values())}")
        print(f"Total PRN's VRAM: {sum(sum(prn_vram) for prn_vram in self.prn.values())}")
        print(f"Total GPU's VRAM: {self.gpu_n*self.gpu_vram}\n")

    def print_prns(self) -> None:
        print("\n=============================")
        print("PRN's VRAM by Type")
        print(self.filename)
        print("=============================")
        for type in self.prn:
            print(f"[{type+1}] : {self.prn[type]}")

    def optimize(self):
        self.gpus = [{'prns': [], 'occupied_vram': 0}]
        current_gpu = 0
        for i in range(len(self.prns)):
            if self.gpus[current_gpu].vram + self.prns[i]['occupied_vram'] > self.gpu_vram:
                self.gpus[current_gpu].prns.append(i)
                self.gpus[current_gpu].vram += self.prns[i]['occupied_vram']
            else:
                current_gpu += 1
                self.gpus.append({'prns': [i], 'occupied_vram': self.prns[i]['vram']})
        
        # Ver se esse sort bombou
        self.gpus.sort(key=lambda x: x['occupied_vram'])
        
    
        def destroy():
            free_prns = self.gpus[:-1].prns
            self.gpus[:-1].pop()
            return free_prns
            
        
        def constroy(free_prns):
            for prn in free_prns:
                
            return
        return

    def optimize_gurobi(self):
        # Parâmetros
        n = self.gpu_n
        m = self.prn_n
        V = self.gpu_vram
        v = []
        t = []
        for prn in self.prns:
            v.append(int(prn["vram"]))
            t.append(int(prn["type"]))

        types = range(self.prn_types_n)

        model = Model("Dog")

        x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
        y = model.addVars(n, len(types), vtype=GRB.BINARY, name="y")

        # Função objetivo: minimizar o número de tipos de PRNs processados
        model.setObjective(y.sum(), GRB.MINIMIZE)

        # Restrição (1): Limite de capacidade de VRAM por GPU
        for i in range(n):
            model.addConstr(sum(x[i, j] * v[j] for j in range(m)) <= V, name=f"VRAM_{i}")

        # Restrição (2): Cada PRN deve ser processada por exatamente uma GPU
        for j in range(m):
            model.addConstr(sum(x[i, j] for i in range(n)) == 1, name=f"AssignPRN_{j}")

        # Restrição (3): Ligação entre x e y
        for i in range(n):
            for j in range(m):
                prn_type_index = types.index(t[j])
                model.addConstr(x[i, j] <= y[i, prn_type_index], name=f"Link_x_y_{i}_{j}")

        model.setParam('TimeLimit', 60)

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            print("\nOptimal Solution Found:")
            print(f"Custo total: {model.ObjVal}")

            print("\nMin PRN\'s assignment by GPU:")
            for i in range(n):
                prns = [j for j in range(m) if x[i, j].X > 0.5]
                print(f"GPU {i + 1}: PRNs {prns}")

            print("\nNumber of PRN\'s types processed by GPU\'s:")
            for i in range(n):
                processed_types = [types[j] for j in range(len(types)) if y [i, j].X > 0.5]
                print(f"GPU {i + 1}: ({len(processed_types)}) Types {processed_types} ")
        else:
            print("\nOptimal solution not found.")
        return
