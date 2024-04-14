import pathlib
import pickle
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
#the above import is for mixed precision training - policies are different ways to do mixed precision training with some example policies being
# max_float16, mixed_bfloat16, etc. 

#for distributed training: values allow you to create and work with distributed values representing the same value across multiple devices or replicas. 
#tend to be used with "Strategies"
try:
  from tensorflow.python.distribute import values
except Exception:
  from google3.third_party.tensorflow.python.distribute import values

tf.tensor = tf.convert_to_tensor
for base in (tf.Tensor, tf.Variable, values.PerReplica):
  base.mean = tf.math.reduce_mean
  base.std = tf.math.reduce_std
  base.var = tf.math.reduce_variance
  base.sum = tf.math.reduce_sum
  base.any = tf.math.reduce_any
  base.all = tf.math.reduce_all
  base.min = tf.math.reduce_min
  base.max = tf.math.reduce_max
  base.abs = tf.math.abs
  base.logsumexp = tf.math.reduce_logsumexp
  base.transpose = tf.transpose
  base.reshape = tf.reshape
  base.astype = tf.cast


# values.PerReplica.dtype = property(lambda self: self.values[0].dtype)

# tf.TensorHandle.__repr__ = lambda x: '<tensor>'
# tf.TensorHandle.__str__ = lambda x: '<tensor>'
# np.set_printoptions(threshold=5, edgeitems=0)

def softargmax(x, beta=1e10):
  # x = tf.convert_to_tensor(x)
  x_range = tf.range(x.shape.as_list()[-1], dtype=x.dtype)
  return tf.reduce_sum(tf.nn.softmax(x*beta) * x_range, axis=-1)


class Module(tf.Module): #the base class for all modules in tensorflow (Dense, Conv2D, etc.)

  def save(self, filename): #for saving tensors and parameters
    values = tf.nest.map_structure(lambda x: x.numpy(), self.variables)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Save checkpoint with {amount} tensors and {count} parameters.')
    with pathlib.Path(filename).open('wb') as f:
      pickle.dump(values, f)

  def load(self, filename): #loading checkpoint
    with pathlib.Path(filename).open('rb') as f:
      values = pickle.load(f)
    amount = len(tf.nest.flatten(values))
    count = int(sum(np.prod(x.shape) for x in tf.nest.flatten(values)))
    print(f'Load checkpoint with {amount} tensors and {count} parameters.')
    tf.nest.map_structure(lambda x, y: x.assign(y), self.variables, values)

  def get(self, name, ctor, *args, **kwargs):
    # Create or get layer by name to avoid mentioning it in the constructor.
    if not hasattr(self, '_modules'):
      self._modules = {}
    if name not in self._modules:
      self._modules[name] = ctor(*args, **kwargs)
    return self._modules[name]


class Optimizer(tf.Module):

  def __init__(
      self, name, lr, eps=1e-4, clip=None, wd=None,
      opt='adam', wd_pattern=r'.*'):
    assert 0 <= wd < 1 #weight decay
    assert not clip or 1 <= clip #grad clip
    self._name = name 
    self._clip = clip
    self._wd = wd
    self._wd_pattern = wd_pattern #interesting.. the pattern of weights to which decay is to be applied
    self._opt = { # a dictionary of optimmizers lol
        'adam': lambda: tf.optimizers.Adam(lr, epsilon=eps),
        'nadam': lambda: tf.optimizers.Nadam(lr, epsilon=eps),
        'adamax': lambda: tf.optimizers.Adamax(lr, epsilon=eps),
        'sgd': lambda: tf.optimizers.SGD(lr),
        'momentum': lambda: tf.optimizers.SGD(lr, 0.9),
    }[opt]()
    self._mixed = (prec.global_policy().compute_dtype == tf.float16) # mixed precision policy dtype
    #NVIDIA GPUs run float16 ops faster than float32s apparently. 
    if self._mixed:
      self._opt = prec.LossScaleOptimizer(self._opt, dynamic=True) # scaling loss coz float16 has a tendency to underflow
    self._once = True #used later in call

  @property
  def variables(self):
    return self._opt.variables()

  def __call__(self, tape, loss, modules):
    #tape is for automatic differentiation: look tf.GradientTape. modules are the modules of the model
    assert loss.dtype is tf.float32, (self._name, loss.dtype)
    assert len(loss.shape) == 0, (self._name, loss.shape)
    metrics = {}

    # Find variables.
    modules = modules if hasattr(modules, '__len__') else (modules,)
    varibs = tf.nest.flatten([module.variables for module in modules]) #get all the vars you need to opt
    count = sum(np.prod(x.shape) for x in varibs)
    if self._once:
      print(f'Found {count} {self._name} parameters.')
      self._once = False
      #print all the layers in each named module along with the parameters per
      # for module in modules:
      #   print(f'{module.__class__.__name__}:')
      #   for var in module.variables:
      #     print(f'  {var.name} {var.shape}')

    # Check loss.
    tf.debugging.check_numerics(loss, self._name + '_loss') # shudnt be nan or inf or something
    metrics[f'{self._name}_loss'] = loss # add loss to metrics

    # Compute scaled gradient.
    if self._mixed:
      with tape:
        loss = self._opt.get_scaled_loss(loss) #record loss for scaling so as to play it back later when computing gradients
    grads = tape.gradient(loss, varibs) #here we go
    if self._mixed:
      grads = self._opt.get_unscaled_gradients(grads) #unscale the gradients
    if self._mixed:
      metrics[f'{self._name}_loss_scale'] = self._opt.loss_scale

    # Distributed sync.
    context = tf.distribute.get_replica_context()

    #only consider grads that are not None
    # grads = [g if g is not None else 0.0 for g in grads]

    if context:
      grads = context.all_reduce('mean', grads) # mean across all devices

    # Gradient clipping.
    norm = tf.linalg.global_norm(grads)
    if not self._mixed:
      tf.debugging.check_numerics(norm, self._name + '_norm')
    if self._clip:
      grads, _ = tf.clip_by_global_norm(grads, self._clip, norm)
    metrics[f'{self._name}_grad_norm'] = norm

    # Weight decay.
    if self._wd:
      self._apply_weight_decay(varibs)

    # Apply gradients.
    self._opt.apply_gradients(
        zip(grads, varibs),
        experimental_aggregate_gradients=False)

    return metrics

  def _apply_weight_decay(self, varibs):
    nontrivial = (self._wd_pattern != r'.*')
    if nontrivial:
      print('Applied weight decay to variables:')
    for var in varibs:
      if re.search(self._wd_pattern, self._name + '/' + var.name):
        if nontrivial:
          print('- ' + self._name + '/' + var.name)
        var.assign((1 - self._wd) * var)
