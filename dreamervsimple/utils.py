import torch
def static_scan(fn, inputs, start, reverse=False):
  last = start
  outputs = [[] for _ in start]
  indices = range(inputs[0].shape[0]) # assuming action is not a dictionary lol -- so this is the shape of the batch
  if reverse:
    indices = reversed(indices)
  for index in indices: # for each item in the batch
    inp = (lambda y: [x[index] for x in y], inputs)
    last = fn(last, inp)
    [o.append(l) for o, l in zip(outputs, last)]
  if reverse:
    outputs = [list(reversed(x)) for x in outputs]
  outputs = [torch.stack(x, 0) for x in outputs]
  return outputs if len(start) == 3 else (outputs[:3], outputs[3:])