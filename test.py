from solver import GreedyDogSolver

dogs = [GreedyDogSolver(f'instances/dog_{i+1}.txt') for i in range(10)]

for dog in dogs:
    #dog.print_instance_info()
    dog.optimize_gurobi(time_limit=1800)
