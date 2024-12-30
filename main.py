import argparse
from solver import GreedyDogSolver

parser = argparse.ArgumentParser(description="Run SoupSolver for a given problem instance.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("output",
                    help="path to output file",
                    type=str)
parser.add_argument("instance",
                    help="path to the problem instance",
                    type=str)
parser.add_argument("-t", metavar="TIME",
                    help="max time in seconds",
                    default=300,
                    type=int)

args = parser.parse_args()

dog = GreedyDogSolver()
dog.solve()

