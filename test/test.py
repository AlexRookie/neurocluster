import platform
print('python: ', platform.python_version())

import sys
print('path: ',sys.executable)

import numpy
print('numpy: ',numpy.__version__)

import tensorflow as tf
print('tensorflow: ',tf.__version__)

a = tf.Variable([1,2,3])
print('tensor a: ',a)