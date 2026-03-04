import inspect

import tensorflow as tf


def main() -> None:
    print("TF version:", tf.__version__)
    print("TF path:", inspect.getfile(tf))
    print("Build info:", tf.sysconfig.get_build_info())

    physical_gpus = tf.config.list_physical_devices("GPU")
    print("Physical GPUs:", physical_gpus)
    print("Logical devices:", tf.config.list_logical_devices())

    device = "/GPU:0" if physical_gpus else "/CPU:0"
    print("Using device:", device)

    with tf.device(device):
        a = tf.random.uniform((128, 128))
        b = tf.random.uniform((128, 128))
        c = tf.matmul(a, b)

    print("Matmul shape:", c.shape)
    print("Matmul device:", c.device)


if __name__ == "__main__":
    main()
