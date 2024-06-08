import numpy as np
import scipy.fftpack as fftpack

def procesar_tableros(labels, nbits):
  grids          = labels.transpose((2,0,1)).astype(np.float32)
  numstates      = 2**nbits
  angleValues    = np.linspace(0,  2, numstates, endpoint=False, dtype=np.float32) # values as they should appear in the tensor imported from matlab
  balancedValues = np.linspace(-1, 1, numstates,                 dtype=np.float32) # values for dct balacing
  all_grids          = []
  all_symbolic_grids = []
  all_balanced_grids = []
  #convert grids to symbolic values in range(numstates) to avoid annoying floating point errors from biting us in the ass
  grids_symbolic     = np.full(grids.shape, -1, np.int8)
  cutoff = 1e-4
  for i,v in enumerate(angleValues):
    mask = np.abs(grids-v)<cutoff
    if (grids_symbolic[mask]!=-1).any():
      raise Exception('there is something wrong with the grids!!!!')
    grids_symbolic[mask] = i
  if (grids_symbolic<0).any() or (grids_symbolic>=numstates).any():
    raise Exception(f'there are non-valid values in the grid!!! Valid values are {values}')
  for k in range(numstates):
    #compute current sym grids
    current_sym = (grids_symbolic+k)%numstates
    #compute current/balanced grids
    current_grids = np.zeros(grids.shape, dtype=np.float32)
    current_balanced_grids = np.zeros(grids.shape, dtype=np.float32)
    for i,(v,b) in enumerate(zip(angleValues, balancedValues)):
      mask = current_sym==i
      current_grids[mask] = v
      current_balanced_grids[mask] = b
    all_grids.append(current_grids)
    all_symbolic_grids.append(current_sym)
    all_balanced_grids.append(current_balanced_grids)
  grids          = np.stack(all_grids,          axis=1)
  balanced_grids = np.stack(all_balanced_grids, axis=1)
  dcts           = np.zeros(grids.shape, dtype=np.float32)
  for k in range(grids.shape[0]):
    for i in range(grids.shape[1]):
      dcts[k,i,:,:]    = fftpack.dctn(balanced_grids[k,i,:,:], norm='ortho')
  means = dcts.reshape(*dcts.shape[:2], -1)
  means = means.mean(axis=-1)
  means = np.argmin(means, axis=-1)
  if len(means.shape)!=1 or means.shape[0]!=grids.shape[0]:
    raise Exception('strange error computing canonical indexes!!!!')
  for i,s in enumerate(means):
    #if not already there, put the canonical instance at position 0 in all tensors replicated for numstates
    if s!=0:
      canonical_grid      = grids[i,s].copy()
      grids[i,s]          = grids[i,0]
      grids[i,0]          = canonical_grid
  return grids[:,0,:,:]
