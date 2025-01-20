import argparse
from solver import GreedyDogSolver

parser = argparse.ArgumentParser(description="Run GreedyDogSolver for a given DOG instance.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("output",
                    help="Path to the output file where the best solution will be saved.",
                    type=str)
parser.add_argument("instance",
                    help="Path to the problem instance file.",
                    type=str)
parser.add_argument("-i", "--iterations",
                    help="Maximum number of iterations.",
                    default=100000,
                    type=int)
parser.add_argument("-t", "--temperature",
                    help="Initial temperature for the solver.",
                    default=0.3,
                    type=float)
parser.add_argument("-s", "--seed",
                    help="Seed for random number generation.",
                    default=None,
                    type=int)
parser.add_argument("-e", "--enhanced",
                    help="Use an enhanced initial solution to improve solver performance.",
                    action="store_true")
parser.add_argument("-p", "--plot",
                    help="Plot initial solution and final solution distribution.",
                    action="store_true")

args = parser.parse_args()

dog = GreedyDogSolver(filename=args.instance)
dog.solve(output_file=args.output, max_iterations=args.iterations, temperature=args.temperature, seed=args.seed, enhanced=args.enhanced, plot=args.plot)
