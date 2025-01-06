import csv
from solver import GreedyDogSolver

# Configuração
dogs = [GreedyDogSolver(f'instances/dog_{i+1}.txt') for i in range(10)]
temperatures = [0.01, 5, 10]
seed = 42
iterations = 1000000
csv_file = 'results.csv'
enhanced = [False, True]

# Inicializa o CSV com cabeçalho
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Instance', 'Initial Solution', 'Best Solution', 'Temperature', 'Iterations', 'Execution Time', 'Enhanced'])

# Resolve e salva os dados no CSV
for i, dog in enumerate(dogs):
    for t in temperatures:
        for e in enhanced:
            # Resolve o problema
            dog.solve(output_file=f'./solutions/dog_{i+1}_1mi_{t}.csv', max_iterations=iterations, temperature=t, seed=seed, enhanced=e)
            # Avalia a solução
            obj = dog.avaluate_solution(dog.gpus)
            # Escreve no CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"dog_{i + 1}", dog.initial_solution_value, dog.best_solution_value, t, iterations, f'{dog.execution_time:.2f}', e])