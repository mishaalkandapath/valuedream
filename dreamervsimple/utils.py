import torch
import optree

def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in start]
  indices = range(inputs[0].shape[0]) # assuming action is not a dictionary lol -- so this is the shape of the batch
  if reverse:
    indices = reversed(indices)
  for index in indices: # for each item in the batch
    inp = optree.tree_map(lambda x: x[index], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, last)]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [torch.stack(x, 0) for x in outputs]
  return outputs if len(start) == 3 else (outputs[:3], outputs[3:])

def calc_conv_shape(in_shape= (3, 64, 64), cnn_depth=48, kernels=(4, 4, 4, 4), stride=2):
  #C H W
  out_depth = 2 ** (len(kernels) - 1) * cnn_depth
  shaper = lambda x, i: int((x - kernels[i])/stride) + 1
  for i in range(len(kernels)):
    if i == 0:
      _, in_height, in_width = in_shape
    else: in_height, in_width = out_height, out_width
    out_height, out_width = shaper(in_height, i), shaper(in_width, i)
  return (2 ** (len(kernels) - 1) * cnn_depth, out_height, out_width)


