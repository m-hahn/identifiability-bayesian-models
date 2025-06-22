import math
import torch


def setEncoding(ENCODING, parameters, grid, MAX_GRID):
   if ENCODING == "WEBER":
     epsilon = math.exp(-1.846029281616211)
     parameters["volume"] = torch.softmax(-torch.log(epsilon + grid), dim=0).log()
   elif ENCODING == "WEBERFIT":
     epsilon = 0.1
     parameters["volume"] = torch.softmax(-torch.log(epsilon + grid), dim=0).log()
   elif ENCODING == "UNIFORM":
     parameters["volume"] = torch.softmax(0*grid, dim=0)
   elif ENCODING == "STEEPPERIODIC":
      assert False
      parameters["volume"] = torch.softmax(2 * (2+torch.sin(15*grid)).log(), dim=0).log()
   elif ENCODING == "UNIMODAL2":
      parameters["volume"] = torch.softmax(-(grid-grid.mean()).pow(2)*1.5, dim=0).log()
   elif ENCODING == "BIMODAL2":
      parameters["volume"] = (torch.softmax(-(grid-grid.min()).pow(2)*1.5, dim=0)+torch.softmax(-(grid-grid.max()).pow(2)*1.5, dim=0)).log()
   else:
     assert False, ENCODING

def setPrior(PRIOR, parameters, grid, MAX_GRID):
   if PRIOR == "LOGNORMAL":
      epsilon = 0.1
      parameters["prior"] = (-((epsilon+grid).log()-(0.72)).pow(2)/(1e-5+math.exp(-1)) - (epsilon+grid).log())
      print(torch.softmax(parameters["prior"], dim=0))
   elif PRIOR == "FROMFIT":
      epsilon = math.exp(-1.846029281616211)
      parameters["prior"] = (-((epsilon+grid).log()-(-0.047955773770809174)).pow(2)/(1e-5+math.exp(-3.6692357063293457)) - (epsilon+grid).log())
      print(torch.softmax(parameters["prior"], dim=0))
   elif PRIOR == "LOGNORMALLEFT":
      epsilon = 0.1
      parameters["prior"] = (-((epsilon+grid).log()-(-0.2)).pow(2)/(1e-5+math.exp(-2.7)) - (epsilon+grid).log())
      print(torch.softmax(parameters["prior"], dim=0))
   elif PRIOR == "BILOGNORMAL":
      epsilon = 0.1
      part1 = (-((epsilon+grid).log()-(-0.1)).pow(2)/(1e-5+math.exp(-2)) - (epsilon+grid).log())
      part2 = (-((epsilon+grid).log()-(0.95)).pow(2)/(1e-5+math.exp(-2)) - (epsilon+grid).log())
      parameters["prior"] = (torch.softmax(part1, dim=0) + torch.softmax(part2, dim=0)).log()
      print(torch.softmax(parameters["prior"], dim=0))
   elif PRIOR == "BILOGNORMALLEFT":
      epsilon = 0.1
      part1 = (-((epsilon+grid).log()-(-0.55)).pow(2)/(1e-5+math.exp(-2.8)) - (epsilon+grid).log())
      part2 = (-((epsilon+grid).log()-(0.2)).pow(2)/(1e-5+math.exp(-2.8)) - (epsilon+grid).log())
      parameters["prior"] = (torch.softmax(part1, dim=0) + torch.softmax(part2, dim=0)).log()
      print(torch.softmax(parameters["prior"], dim=0))
   elif PRIOR == "BIFROMFIT":
      epsilon = math.exp(-1.846029281616211)
      part1 = (-((epsilon+grid).log()-(-0.45)).pow(2)/(1e-5+math.exp(-3.6692357063293457)) - (epsilon+grid).log())
      part2 = (-((epsilon+grid).log()-(0.1)).pow(2)/(1e-5+math.exp(-3.6692357063293457)) - (epsilon+grid).log())
      parameters["prior"] = (torch.softmax(part1, dim=0) + torch.softmax(part2, dim=0)).log()
      print(torch.softmax(parameters["prior"], dim=0))
   elif PRIOR == "BIMODAL2":
      parameters["prior"] = (torch.softmax(-(grid-grid.min()).pow(2)*1.5, dim=0)+torch.softmax(-(grid-grid.max()).pow(2)*1.5, dim=0)).log()
   elif PRIOR == "UNIMODAL2":
      parameters["prior"] = torch.softmax(-(grid-grid.mean()).pow(2)*1.5, dim=0).log()
   elif PRIOR == "UNIFORM":
     parameters["prior"] = torch.softmax(0*grid, dim=0)
   else:
      assert False, PRIOR
   
   
