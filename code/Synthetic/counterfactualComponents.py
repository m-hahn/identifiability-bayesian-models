import math
import torch
import util

def setPrior(PRIOR, parameters, grid, MAX_GRID):
  if PRIOR == "UNIFORM":
     parameters["prior"] = 0*grid
  elif PRIOR == "PERIODIC":
     parameters["prior"] = (2-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif PRIOR == "SHIFTED":
     parameters["prior"] = (2-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
  elif PRIOR == "STEEPSHIFTED":
     parameters["prior"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
  elif PRIOR == "STEEPPERIODIC":
     parameters["prior"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif PRIOR == "SQUAREPERIODIC":
     parameters["prior"] = 2 * (2-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif PRIOR == "SQUARESTEEPPERIODIC":
     parameters["prior"] = 2 * (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif PRIOR == "SQRTSTEEPPERIODIC":
     parameters["prior"] = .5 * (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif PRIOR == "UNIMODAL2":
     parameters["prior"] = .2 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
  elif PRIOR == "UNIMODAL":
     parameters["prior"] = .5 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
  elif PRIOR == "PIECEWISECONSTANT":
     parameters["prior"] = torch.cos(grid/MAX_GRID*2*math.pi+math.pi).sign()
  elif PRIOR == "FITTED":
     pass
  elif PRIOR.startswith("FOURIER1_"):
    _, seed = PRIOR.split("_")
    import random
    rstate = random.Random(int(seed))
    frequencies = util.MakeLongTensor(range(5))
    sines = torch.sin(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    cosines = torch.cos(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    basis = torch.cat([sines, cosines], dim=0)
    coefficients = util.MakeFloatTensor([rstate.random()-0.5 for _ in range(10)]) / (1+torch.cat([frequencies, frequencies], dim=0))
    parameters["prior"] = (basis*coefficients.unsqueeze(1)).sum(dim=0)
  elif PRIOR.startswith("FOURIER_"):
    _, seed = PRIOR.split("_")
    import random
    rstate = random.Random(int(seed))
    frequencies = util.MakeLongTensor(range(5))
    sines = torch.sin(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    cosines = torch.cos(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    basis = torch.cat([sines, cosines], dim=0)
    coefficients = util.MakeFloatTensor([rstate.random()-0.5 for _ in range(10)])  #/ torch.cat([frequencies, frequencies], dim=0).clamp(min=1)
    parameters["prior"] = (basis*coefficients.unsqueeze(1)).sum(dim=0)
  #  figure, axis = plt.subplots(1, 1)
  #  axis.scatter(grid.detach(), torch.softmax(parameters["prior"].detach(), dim=0))
  #  axis.scatter(grid.detach(), 0*grid.detach())
  #  plt.show()
  #  plt.close()
  else:
     assert False, PRIOR
 
def setEncoding(ENCODING, parameters, grid, MAX_GRID): 
  if ENCODING == "PERIODIC":
     parameters["volume"] = (2-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif ENCODING == "SHIFTED":
     parameters["volume"] = (2-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
  elif ENCODING == "STEEPSHIFTED":
     parameters["volume"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi+math.pi/2).abs()).log()
  elif ENCODING == "UNIFORM":
     parameters["volume"] = 0*grid
  elif ENCODING == "UNIMODAL2":
     parameters["volume"] = .2 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
  elif ENCODING == "UNIMODAL":
     parameters["volume"] = .5 * torch.cos(grid/MAX_GRID*2*math.pi+math.pi)
  elif ENCODING == "FITTED":
     pass
  elif ENCODING == "STEEPPERIODIC":
     parameters["volume"] = (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif ENCODING.startswith("STEEPPERIODICa"):
     exponent_integer, exponent_fractional = ENCODING.split("a", 1)[1].split("b", 1)
     parameters["volume"] = float(f"{exponent_integer}.{exponent_fractional}") * (1.5-torch.sin(grid/MAX_GRID*2*math.pi).abs()).log()
  elif ENCODING.startswith("FOURIERa"):
     exponent_string, seed = ENCODING.split("_", 1)
     exponent_integer, exponent_fractional = exponent_string.split("a", 1)[1].split("b", 1)
     exponent = float(f"{exponent_integer}.{exponent_fractional}")
     import random
     rstate = random.Random(int(seed))
     frequencies = util.MakeLongTensor(range(5))
     sines = torch.sin(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
     cosines = torch.cos(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
     basis = torch.cat([sines, cosines], dim=0)
     coefficients = util.MakeFloatTensor([rstate.random()-0.5 for _ in range(10)])
     parameters["volume"] = exponent * (basis*coefficients.unsqueeze(1)).sum(dim=0)
  elif ENCODING == "PIECEWISECONSTANT":
     parameters["volume"] = torch.cos(grid/MAX_GRID*2*math.pi+math.pi).sign()
  elif ENCODING.startswith("FOURIER1_"):
    _, seed = ENCODING.split("_")
    import random
    rstate = random.Random(int(seed))
    frequencies = util.MakeLongTensor(range(5))
    sines = torch.sin(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    cosines = torch.cos(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    basis = torch.cat([sines, cosines], dim=0)
    coefficients = util.MakeFloatTensor([rstate.random()-0.5 for _ in range(10)])  / (1+torch.cat([frequencies, frequencies], dim=0))
    parameters["volume"] = (basis*coefficients.unsqueeze(1)).sum(dim=0)
  elif ENCODING.startswith("FOURIER_"):
    _, seed = ENCODING.split("_")
    import random
    rstate = random.Random(int(seed))
    frequencies = util.MakeLongTensor(range(5))
    sines = torch.sin(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    cosines = torch.cos(grid.view(1,-1)*frequencies.view(-1,1) * 2*math.pi/MAX_GRID)
    basis = torch.cat([sines, cosines], dim=0)
    coefficients = util.MakeFloatTensor([rstate.random()-0.5 for _ in range(10)])  #/ torch.cat([frequencies, frequencies], dim=0).clamp(min=1)
    parameters["volume"] = (basis*coefficients.unsqueeze(1)).sum(dim=0)
  #  figure, axis = plt.subplots(1, 1)
  #  axis.scatter(grid.detach(), torch.softmax(parameters["prior"].detach(), dim=0))
  #  axis.scatter(grid.detach(), torch.softmax(parameters["volume"].detach(), dim=0))
  #  axis.scatter(grid.detach(), 0*grid.detach())
  #  plt.show()
  #  plt.close()
  else:
     assert False
  
  
