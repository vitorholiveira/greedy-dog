import argparse
from solver import GreedyDogSolver

parser = argparse.ArgumentParser(description="Run Gurobi Optimizer for a given DOG instance.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("output",
                    help="Path to the output file where the best solution will be saved.",
                    type=str)
parser.add_argument("instance",
                    help="Path to the problem instance file.",
                    type=str)
parser.add_argument("-t", "--time",
                    help="Time limit for optimization.",
                    default=100000,
                    type=int)

args = parser.parse_args()

dog = GreedyDogSolver(filename=args.instance)
dog.optimize_glpk(output_file=args.output)
