# Advanced Settings and Usages

This page provides the instructions to run the two other distilled settings we explored in the paper, as well as lists details on how to customize our code, and perform testing.

## Adaptation Setting

### Unknown pre-trained weights
We use `MNIST -> USPS` as example. Commands for other adaptation cases are similar.

1. Train many networks on `MNIST`.

    ```sh
    # 200 networks for training
    python main.py --mode train --dataset MNIST --arch LeNet --n_nets 200 \
        --epochs 40 --decay_epochs 20 --lr 2e-4

    # 20 networks for testing
    python main.py --mode train --dataset MNIST --arch LeNet --n_nets 20 \
        --epochs 40 --decay_epochs 20 --lr 2e-4 --phase test
    ```

2. Train distilled images on these 200 networks to work well on `USPS`. At each iteration, sample 4 of them to train.

    ```sh
    python main.py --mode distill_adapt --source_dataset MNIST --dataset USPS \
        --arch LeNet --train_nets_type loaded --n_nets 200 --sample_n_nets 4 \
        --test_nets_type loaded --test_n_nets 20
    ```

    `test_n_nets` here can be less than the number of testing networks trained in the previous step since these are only used to monitor results during training. In full evaluation time, all networks can be used. See [the Testing section](#testing) below for details.

See [the Distributed Training section](#distributed-training) on using much more networks, e.g., 2000 networks for training in our experiments in paper.

### Fixed known pre-trained weights

Similar as above, but now we only need to train one network, and then train distilled images only on it. E.g.,

```sh
# train 1 network
python main.py --mode train --dataset MNIST --arch LeNet --n_nets 1 \
    --epochs 40 --decay_epochs 20 --lr 2e-4

# train distilled images
python main.py --mode distill_adapt --source_dataset MNIST --dataset USPS \
    --arch LeNet --train_nets_type loaded --n_nets 1 \
    --test_nets_type same_as_train
```

### AlexNet with fixed known weights pre-trained on ImageNet

Here we use the same mode as the basic setting, but use a special initialization method `"imagenet_pretrained"`.

For example, the follow command adapts it to the `PASCAL_VOC` dataset.

```
python main.py --mode distill_basic --dataset PASCAL_VOC --arch AlexNet \
    --distill_steps 1 --distill_epochs 3 --init imagenet_pretrained \
    --train_nets_type known_init --test_nets_type same_as_train \
    --epochs 200 --decay_epochs 30 --distill_lr 0.001
```

Use `--dataset CUB200` to adapt to the `CUB200` dataset.

## Malicious Attack Setting

We use `Cifar10` as example.

1. Similar to the adaptation setting, we start with training many networks.

    ```sh
    # 200 networks for training
    python main.py --mode train --dataset Cifar10 --arch AlexCifarNet --n_nets 2000 \
        --epochs 40 --decay_epochs 20 --lr 2e-4

    # 200 networks for testing
    python main.py --mode train --dataset Cifar10 --arch AlexCifarNet --n_nets 200 \
        --epochs 40 --decay_epochs 20 --lr 2e-4 --phase test
    ```

2. Distill for the malicious attack objective so that these well-optimized networks will misclassify a certain class (`attack_class`) to another  (`target_class`), after training on the distilled images.

    ```sh
    python main.py --mode distill_attack --dataset Cifar10 --arch AlexCifarNet \
        --train_nets_type loaded --n_nets 2000 --sample_n_nets 4 \
        --test_nets_type loaded --test_n_nets 20 \
        --attack_class 0 --target_class 1 --lr 0.02
    ```

See [the Distributed Training section](#distributed-training) on using much more networks, e.g., 2000 networks for training in our experiments in paper.

## Distributed Training

We often need to load multiple networks into GPU memory (e.g., for results presented in the paper, we use 2000 networks for training in adaptation and malicious attack settings). A single GPU can not hold all these networks. In such cases, we can use NCCL distributed training specifying a `world_size` larger than `1`. Then, you need to start `world_size` processes with identical arguments except `device_id`, but each with a different environmental variable `RANK` representing the process rank in `[0, 1, ..., world_size - 1]`.

There are two ways to initialize a process group in PyTorch for distributed training:

1. TCP init. Specify environmental variables `MASTER_ADDR` and `MASTER_PORT`, representing an accessible port from all ranks.

2. File system init. Specify environmental variable `INIT_FILE`, representing a file handle accessible from all ranks.

For example,

1. These commands start 2 processes that each trains 1000 networks on a different GPU (2000 in total):

    ```sh
    # rank 0: gpu 0, train [0, 1000)
    env RANK=0 INIT_FILE=/tmp/distill_init \
    python main.py --mode train --dataset MNIST --arch LeNet --n_nets 2000 \
        --epochs 40 --decay_epochs 20 --lr 2e-4 --world_size 2 --device_id 0

    # rank 1: gpu 1, train [1000, 1000)
    env RANK=1 INIT_FILE=/tmp/distill_init \
    python main.py --mode train --dataset MNIST --arch LeNet --n_nets 2000 \
        --epochs 40 --decay_epochs 20 --lr 2e-4 --world_size 2 --device_id 1
    ```

2. These commands start 4 processes that collectively train distilled images for 2000 pre-trained networks. Each process loads 500 networks on a different GPU, and samples 1 network in each iteration:

    ```sh
    # rank 0: gpu 0, load [0, 500)
    env RANK=0 MASTER_ADDR=XXXXX MASTER_ADDR=23456 \
    python main.py --mode distill_adapt --source_dataset MNIST --dataset USPS \
        --arch LeNet --train_nets_type loaded --n_nets 2000 --sample_n_nets 4 \
        --test_nets_type loaded --test_n_nets 20 --world_size 4 --device_id 0

    # rank 1: gpu 1, load [500, 1000)
    env RANK=1 MASTER_ADDR=XXXXX MASTER_ADDR=23456 \
    python main.py --mode distill_adapt --source_dataset MNIST --dataset USPS \
        --arch LeNet --train_nets_type loaded --n_nets 2000 --sample_n_nets 4 \
        --test_nets_type loaded --test_n_nets 20 --world_size 4 --device_id 1

    # rank 2: gpu 2, load [1000, 1500)
    env RANK=2 MASTER_ADDR=XXXXX MASTER_ADDR=23456 \
    python main.py ---mode distill_adapt --source_dataset MNIST --dataset USPS \
        --arch LeNet --train_nets_type loaded --n_nets 2000 --sample_n_nets 4 \
        --test_nets_type loaded --test_n_nets 20 --world_size 4 --device_id 2

    # rank 3: gpu 3, load [1500, 2000)
    env RANK=3 MASTER_ADDR=XXXXX MASTER_ADDR=23456 \
    python main.py --mode distill_adapt --source_dataset MNIST --dataset USPS \
        --arch LeNet --train_nets_type loaded --n_nets 2000 --sample_n_nets 4 \
        --test_nets_type loaded --test_n_nets 20 --world_size 4 --device_id 3
    ```

Distributed training works not only for using multiple GPUs within a single node, but also for training using multiple nodes within a cluster.

### Testing

Using `--phase test`, we can evaluate our trained distilled images and various baselines by specifying the following options:

+ `test_distilled_images`: Source of distilled images to be evaluated. This must be one of `"loaded"`, `"random_train"`, `"average_train"`, and `"kmeans_train"`, specifying whether to load the trained distilled images, or to compute images from training set. Default: `"loaded"`.
+ `test_distilled_lrs`: Learning rates used to evaluate the distilled images. This must be one of `"loaded"`, `"fix [lr]"`, and `"nearest_neighbor [k] [p]"`. `"fix"` will use constant lr for all steps. `"nearest_neighbor"` will instead use the distilled images for `k`-nearest neighbor classification using `p`-norm. Default: `"loaded"`.
+ `test_n_runs`: Number of times to run the entire evaluation process (i.e., constructing distilled images and evaluate). This is useful when using stochastic methods to construct distilled images, e.g., `"random_train"` and `"kmeans_train"`. Default: `1`.
+ `test_n_nets`: Number of test networks used in each run. Default: `1`.
+ `test_distill_epochs`: Number of epochs to apply distilled images. If `None`, this is set to equal to `distill_epochs` used for training. Default `None`.
+ `test_optimize_n_runs`: For stochastic methods to construct distilled images, setting this to a non-`None` value will optimize the obtained distilled images by evaluating them on `test_niter` batches of **training** images, and picking the best `test_n_runs` out of `test_optimize_n_runs` total sets of distilled images. Default: `None`.
+ `test_optimize_n_nets`: Number of training networks used to optimize distilled images. Only meaningful if `test_optimize_n_runs` is not `None`. Default: `20`.

For example,

1. To evaluate the trained distilled images applied over 10 epochs for `MNIST -> USPS` adaptation setting with unknown initialization on 200 networks:

    ```sh
    python main.py --mode distill_adapt --source_dataset MNIST --dataset USPS \
        --arch LeNet --train_nets_type loaded --n_nets 200 --sample_n_nets 4 \
        --phase test --test_nets_type loaded --test_n_nets 200 \
        --test_distilled_images loaded --test_distilled_lrs loaded \
        --test_distill_epochs 10
    ```

2. To evaluate using optimized random training as distilled images with fixed 0.3 learning rate for basic `MNIST` distillation setting on 200 networks:

    ```sh
    python main.py --mode distill_basic --dataset MNIST --arch LeNet \
        --phase test --train_nets_type unknown_init --test_nets_type   unknown_init \
        --test_distilled_images random_train --test_distilled_lrs fix 0.3 \
        --test_n_nets 200 --test_n_runs 20 \
        --test_optimize_n_runs 50 --test_optimize_n_nets 20
    ```

## Recommended Settings to Train Pre-trained Weights

To obtain the pre-trained weights (`--mode train`), we recommend using
+ `--epochs 40 --decay_epochs 20 --lr 2e-4` for `MNIST` with `LeNet`,
+ `--epochs 130 --decay_epochs 40 --lr 2e-4` for `USPS` with `LeNet`,
+ `--epochs 65 --decay_epochs 20 --lr 2e-4` for `SVHN` with `LeNet`,
+ `--epochs 50 --decay_epochs 7 --lr 1e-3` for `Cifar10` with `AlexCifarNet`.

## Useful Arguments

Below we list some of the options you may want to tune:

+ `distill_steps`: Number of gradient steps in applying distilled images. Each step is associated with a new batch of distilled images, so this also affects the total number of images. Default: `10`.
+ `distill_epochs`: Number of passes to cycle over the gradient steps. E.g., with `distill_steps=10` and `distill_epochs=3`, the images of `10` steps are iterated over `3` times, leading to a total of `30` gradient steps. This does not change the total number of distilled images. Default: `3`.
+ `distilled_images_per_class_per_step`: Number of distilled images per class in each step. Default: `1`.
+ `distill_lr`: Initial value of the trained learning rates for distillation. Default: `0.001`.
+ `train_nets_type`: How the initial weights for training are obtained. This must be one of `"unknown_init"` (randomly initialize weights at every iteration), `"known_init"` (initialize weights once before training and keep fixed throughout training), `"loaded"` (weights loaded from disk). Default: `"unknown_init"`.
+ `n_nets`: Number of networks available to train the distilled images *in each iteration*. E.g., with `train_nets_type="unknown_init"` and `n_nets=4`, each training iteration samples 4 new sets of initial weights. Default: `4`.
+ `sample_n_nets`: Number of networks subsampled from `n_nets` networks for training. This option is useful when training for pre-trained weights. E.g., with `train_nets_type="loaded"`, `n_nets=2000` and `sample_n_nets=4`, in each iteration, 4 out of 2000 loaded networks will be randomly selected for training. Default: same as `n_nets`.
+ `test_nets_type`: How the initial weights for testing are obtained. This must be one of `"unknown_init"`, `"same_as_train"` (same weights for training are used in testing), `"loaded"`. Default: `"unknown_init"`.
+ `init`: Initialization method to sample the initial weight. This must be one of `"xavier"`, `"xavier_unif"`, `"kaiming"`, `"kaiming_out"`, `"orthogonal"`, `"default"`, and `"imagenet_pretrained"`. `"default"` uses the default initialization method in PyTorch. `"imagenet_pretrained"` only works with `AlexNet` and loads a particular set of weights pre-trained on ImageNet. Others call corresponding initialization methods. See `init_weights` function in `networks/utils.py` for details. Default: `"xavier"`.
+ `init_param`: Parameters used for the used initialization method, e.g., `gain` argument for `"xavier"`. See `init_weights` function in `networks/utils.py` for details. Default: `1`.
+ `device_id`: The device index used in this training process. If negative, CPU is used. Default: `0`.

## Correctness Test

We include tests checking the correctness of our custom gradient computation. You may use `python test_train_distilled_image.py` to run them. You can optionally append `-v` to enable verbose mode, which prints the numerical gradient test details.
