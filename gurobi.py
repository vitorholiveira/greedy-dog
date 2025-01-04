from gurobipy import Model, GRB
from instance import DogInstance

def solve_dog(instance: DogInstance):

    # Parâmetros
    n = instance.gpu_n
    m = instance.prn_n
    V = instance.gpu_vram
    v = []
    t = []
    for prn in instance.prns:
        v.append(int(prn["vram"]))
        t.append(int(prn["type"]))

    types = range(instance.prn_types_n)

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
            processed_types = [types[j] for j in range(len(types)) if y[i, j].X > 0.5]
            print(f"GPU {i + 1}: ({len(processed_types)}) Types {processed_types} ")
    else:
        print("\nOptimal solution not found.")
