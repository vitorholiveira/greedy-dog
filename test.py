from solver import GreedyDogSolver

dogs = [GreedyDogSolver(f'instances/dog_10.txt')]

for dog in dogs:
    dog.print_instance_info()
    dog.solve(steroids=True)
