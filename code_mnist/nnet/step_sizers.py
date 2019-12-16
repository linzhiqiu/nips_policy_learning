
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


def const_step(init_step, global_step):
  return ops.convert_to_tensor(init_step)


def sqrt_step(init_step, global_step):
  return math_ops.multiply(ops.convert_to_tensor(init_step),
                           math_ops.rsqrt(
                               math_ops.to_float(math_ops.add(1, global_step))))

def exp_stair_step(init_step, global_step,
                   steps_per_epoch=1, delay=1, base=0.5):
  exponent = math_ops.maximum(0, math_ops.add(
      1-delay, math_ops.floor_div(global_step, steps_per_epoch)))
  step_size = init_step * math_ops.pow(base, math_ops.to_float(exponent))
  return ops.convert_to_tensor(step_size)
