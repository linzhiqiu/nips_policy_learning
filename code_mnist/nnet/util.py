
import tensorflow as tf

def on_indices(x, a):
  """useful utility function"""
  row_indices = tf.range(tf.shape(a)[0])
  idx = tf.stack([row_indices, a], axis=1)
  return tf.gather_nd(x, idx)

def copy_model_vars(source_name, target_model):
  copy_ops = []
  target_name = target_model.name
  target_vars = tf.trainable_variables(scope=target_name)
  source_vars = tf.trainable_variables(scope=source_name)
  for target_var in target_vars:
    source_var_name = target_var.name.replace(target_name, source_name)
    source_var = [var for var in source_vars if var.name == source_var_name][0]
    copy_ops.append(tf.assign(target_var, source_var.value()))
  return copy_ops

