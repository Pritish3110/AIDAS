import tensorflow as tf
print("CUDA version:", tf.sysconfig.get_build_info()['cuda_version'])
print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])
print("GPUs:", tf.config.list_physical_devices('GPU'))
