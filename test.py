from instance import DogInstance
import gurobi as gp

dogs = [DogInstance(f'instances/dog_{i+1}.txt') for i in range(10)]

for dog in dogs:
    dog.print_prns()
    gp.solve_dog(dog)
