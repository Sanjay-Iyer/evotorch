import torch
from torch import nn
from evotorch import Problem
from evotorch.logging import StdOutLogger
from evotorch.algorithms import SNES
from evotorch.algorithms import CEM
from evotorch.algorithms import GeneticAlgorithm
from evotorch.algorithms import XNES
from evotorch.algorithms import GeneticAlgorithm
from evotorch.decorators import vectorized
from evotorch.operators import TwoPointCrossOver
from evotorch.operators import OnePointCrossOver, GaussianMutation
from evotorch.operators import SimulatedBinaryCrossOver
from evotorch.operators import PolynomialMutation
import numpy as np
import random

# #prompt for output
def prompt(parameters: torch.Tensor) -> torch.Tensor:
    #prompt user for input from jupyter notebook
    print('1-capillaryvoltage, 2-tubelensvoltage')
    #for loop to print out parameters their representive string 
    for i in range(len(parameters)):
        print('Please run instrument with parameters {}'.format(parameters))
    #  print('Please run instrument with parameters {}'.format(parameters))

    #print('Please run instrument with parameters {}'.format(parameters))
    return torch.tensor(float(input("Enter a number: ")))


#1 sheath_gas = 30 #range? units?
#2 aux_gas = 10 #range? units?
#3 sweep_gas = 0,5 #units?
#4 spray_voltage = 3,5 #kV
#5 capillary_temperature = 275 #range? celcius
#6 capillary_voltage = 10 #range? volts
#7 tube_lens_voltage = 40 #range? volts
#8 flow rate 5 - 50 ul/hr usually 15-20 
problem = Problem(
    'max',
    prompt,
   # fitness,
    solution_length = 2,
    initial_bounds=([10.1,41.7],[10.1,41.7]),
    bounds=(
    [2,2],
    [200,400]
            ),
    dtype=torch.float32,
    #device='cpu'
)
problem
#1 sheath_gas = 30 #range? units?
#2 aux_gas = 10 #range? units?
#3 sweep_gas = 0,5 #units?
#4 spray_voltage = 3,5 #kV
#5 capillary_temperature = 275 #range? celcius
#6 capillary_voltage = 10 #range? volts 2-200
#7 tube_lens_voltage = 40 #range? volts 
#8 flow rate 5 - 50 ul/hr usually 15-20 
#Only positive ion mode 
#negative ion mode changes all the voltages to negative 
#ESI or APCI 
#APIC today more universial 
def my_own_gaussian_mutation(x: torch.Tensor) -> torch.Tensor:
    # The default GaussianMutation of EvoTorch does not (yet) support different standard deviation values
    # per variable. However, we can define our own mutation operator which adds noise of different magnitudes
    # to different variables, like in this example:
    [_, solution_length] = x.shape
    dtype = x.dtype
    device = x.device

    # Generate Gaussian noise where each column has its own magnitude
    noise = (
        torch.randn(solution_length, dtype=dtype, device=device)
        * torch.tensor([5,5], dtype=dtype, device=device)
    )

    return x + noise
searcher = GeneticAlgorithm(
    problem,
    popsize=1,
    operators=[my_own_gaussian_mutation]
       #TwoPointCrossOver(problem, tournament_size=1)
   # ],
)

_ = StdOutLogger(searcher, interval=1)
count = 0
searcher.run(num_generations=10)
list(searcher.population[:10])