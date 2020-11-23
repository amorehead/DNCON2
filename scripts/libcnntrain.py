#!/usr/bin/python
# Badri Adhikari, 6-15-2017
# Alex Morehead, 10-25-2020
# Subroutines for prediction

import math
import os
import sys

import numpy as np
import wandb
from keras_applications import get_submodules_from_kwargs
from keras_applications.imagenet_utils import _obtain_input_shape
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from wandb.keras import WandbCallback

# region Global Variables

# Init wandb
wandb.init(project="dncon2")

epsilon = keras.backend.epsilon()

BASE_WEIGHT_URL = ('https://github.com/fchollet/deep-learning-models/'
                   'releases/download/v0.7/')
backend = None
# layers = None
models = None
keras_utils = None


# endregion


# region Utility Functions
def print_feature_summary(X):
    print("FeatID         Avg        Med        Max        Sum        Avg[30]    Med[30]    Max[30]    Sum[30]")
    for ii in range(0, len(X[0, 0, 0, :])):
        (m, s, a, d) = (
            X[0, :, :, ii].flatten().max(),
            X[0, :, :, ii].flatten().sum(),
            X[0, :, :, ii].flatten().mean(),
            np.median(X[0, :, :, ii].flatten()),
        )
        (m30, s30, a30, d30) = (
            X[0, 30, :, ii].flatten().max(),
            X[0, 30, :, ii].flatten().sum(),
            X[0, 30, :, ii].flatten().mean(),
            np.median(X[0, 30, :, ii].flatten()),
        )
        print(" Feat%2s %10.4f %10.4f %10.4f %10.1f     %10.4f %10.4f %10.4f %10.4f" % (
            ii,
            a,
            d,
            m,
            s,
            a30,
            d30,
            m30,
            s30,
        ))


def get_x_from_this_file(feature_file):
    L = 0
    with open(feature_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            L = line.strip().split()
            L = int(round(math.exp(float(L[0]))))
            break
    x = getX(feature_file, L)
    F = len(x[0, 0, :])
    X = np.zeros((1, L, L, F))
    X[0, :, :, :] = x
    return X


def prediction2rr(P, fileRR):
    print("Writing RR file " + fileRR)
    L = int(math.sqrt(len(P)))
    PM = P.reshape(L, L)
    rr = open(fileRR, "w")
    for i in range(0, L):
        for j in range(i, L):
            if abs(i - j) < 1:
                continue
            rr.write("%i %i 0 8 %.5f\n" % (i + 1, j + 1, PM[i][j]))
    rr.close()


# Model architectures / Layers information
def read_model_arch(file_config):
    if not os.path.isfile(file_config):
        print('Error! Could not find config file ' + file_config)
        sys.exit(1)
    lyrs = {}
    with open(file_config) as f:
        for line in f:
            if line.startswith('#'):
                continue
            if len(line) < 2:
                continue
            cols = line.strip().split()
            if len(cols) != 7:
                print('Error! Config file ' + file_config + ' line ' + line + '??')
                sys.exit(1)
            lyrs[cols[0]] = cols[1] + ' ' + cols[2] + ' ' + cols[3] + ' ' + cols[4] + ' ' + cols[5] + ' ' + cols[6]
    print('')
    print('Read model architecture:')
    for k, v in sorted(lyrs.items()):
        print(k + ' : ' + v)
    print('')
    return lyrs


# Feature file that has 0D, 1D, and 2D features (L is the first feature)
# Output size (a little >= L) to which all the features will be rolled up to as 2D features
def getX(feature_file, L_MAX):
    # calculate the length of the protein (the first feature)
    reject_list = []
    reject_list.append("# PSSM")
    reject_list.append("# AA composition")
    L = 0
    with open(feature_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            L = line.strip().split()
            L = int(round(math.exp(float(L[0]))))
            break
    Data = []
    with open(feature_file) as f:
        accept_flag = 1
        for line in f:
            if line.startswith("#"):
                if line.strip() in reject_list:
                    accept_flag = 0
                else:
                    accept_flag = 1
                continue
            if accept_flag == 0:
                continue
            if line.startswith("#"):
                continue
            this_line = line.strip().split()
            if len(this_line) == 0:
                continue
            if len(this_line) == 1:
                # 0D feature
                feature2D = np.zeros((L, L))
                feature2D[:, :] = float(this_line[0])
                Data.append(feature2D)
            elif len(this_line) == L:
                # 1D feature
                feature2D1 = np.zeros((L, L))
                feature2D2 = np.zeros((L, L))
                for i in range(0, L):
                    feature2D1[i, :] = float(this_line[i])
                    feature2D2[:, i] = float(this_line[i])
                Data.append(feature2D1)
                Data.append(feature2D2)
            elif len(this_line) == L * L:
                # 2D feature
                feature2D = np.asarray(this_line).reshape(L, L)
                Data.append(feature2D)
            else:
                print(line)
                print("Error!! Unknown length of feature in !!" + feature_file)
                print(
                    "Expected length 0, "
                    + str(L)
                    + ", or "
                    + str(L * L)
                    + " - Found "
                    + str(len(this_line))
                )
                sys.exit()
    F = len(Data)
    X = np.zeros((L_MAX, L_MAX, F))
    for i in range(0, F):
        X[0:L, 0:L, i] = Data[i]
    return X, L


# endregion


# region Original Architecture
def build_orig_model_for_this_input_shape(model_arch, input_shape=None):
    """Old DNCON2 Architecture with Dropout"""
    layer = 1
    model = keras.Sequential()
    while True:
        if not ("layer" + str(layer)) in model_arch:
            break
        parameters = model_arch["layer" + str(layer)]
        cols = parameters.split()
        num_kernels = int(cols[0])
        filter_size = int(cols[1])
        b_norm_flag = cols[2]
        max_pool_flag = cols[3]  # Disabled for dimension mismatch
        activ_funct = cols[4]
        dropout_rate = float(cols[5])  # Disabled for dimension mismatch
        if layer == 1:
            model.add(
                layers.Conv2D(filters=num_kernels,
                              kernel_size=filter_size,
                              padding="same",
                              input_shape=input_shape))
        else:
            model.add(
                layers.Conv2D(filters=num_kernels,
                              kernel_size=filter_size,
                              padding="same",
                              input_shape=input_shape))
        if b_norm_flag == "1":
            model.add(layers.BatchNormalization())
        # if max_pool_flag == "1":
        #     model.add(layers.MaxPool2D())
        model.add(layers.Activation(activ_funct))
        # if dropout_rate > 0.0:
        #     model.add(layers.Dropout(dropout_rate))
        layer += 1
    model.add(layers.Flatten())
    return model


# endregion


# region Resnet-50 Architecture
def res_conv(x, s, filters):
    """Here, the input size changes"""
    x_skip = x
    f1, f2 = filters

    # First block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    # When s = 2 then it is like downsizing the feature map
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # Second block
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # Third block
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)

    # Shortcut
    x_skip = layers.Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
                           kernel_regularizer=regularizers.l2(0.001))(x_skip)
    x_skip = layers.BatchNormalization()(x_skip)

    # Add
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activations.relu)(x)

    return x


def res_identity(x, filters):
    """
    ResNet block where dimensions do not change.
    The skip connection is just a simple identity connection.
    We will have 3 blocks, and then input will be added.
    """

    x_skip = x  # Will be used for addition with the residual block
    f1, f2 = filters

    # First block
    x = layers.Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # Second block # bottleneck (but size kept same with padding)
    x = layers.Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)

    # Third block activation used after adding the input
    x = layers.Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                      kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # Add the input
    x = layers.Add()([x, x_skip])
    x = layers.Activation(activations.relu)(x)

    return x


def build_resnet_model_for_this_input_shape(input_shape, num_of_classes):
    input_layer = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3))(input_layer)

    # 1st stage
    # Here, we perform max pooling
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activations.relu)(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2nd stage
    # From here on, only convolution block and identity blocks, no pooling
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage
    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # End with average pooling and a dense connection
    x = layers.AveragePooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(num_of_classes, activation='softmax', kernel_initializer='he_normal')(x)  # Multi-class

    # Define the model
    model = keras.models.Model(inputs=input_layer, outputs=x, name='Resnet50')

    return model


# endregion


# region Inception-Resnet-V2 Architecture
def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = layers.Conv2D(filters,
                      kernel_size,
                      strides=strides,
                      padding=padding,
                      use_bias=use_bias,
                      name=name)(x)
    if not use_bias:
        bn_axis = 1 if backend.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = layers.BatchNormalization(axis=bn_axis,
                                      scale=False,
                                      name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = layers.Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 128, 1)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7])
        branch_1 = conv2d_bn(branch_1, 192, [7, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(x, 192, 1)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3])
        branch_1 = conv2d_bn(branch_1, 256, [3, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    mixed = layers.Concatenate(
        axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   backend.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   name=block_name + '_conv')

    x = layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=backend.int_shape(x)[1:],
                      arguments={'scale': scale},
                      name=block_name)([x, up])
    if activation is not None:
        x = layers.Activation(activation, name=block_name + '_ac')(x)
    return x


def build_inception_resnet_v2_model_for_this_input_shape(include_top=True,
                                                         weights='imagenet',
                                                         input_tensor=None,
                                                         input_shape=None,
                                                         pooling=None,
                                                         classes=2,
                                                         **kwargs):
    """Instantiates the Inception-ResNet v2 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # New DNCON2 Inception-Resnet-V2 Architecture #
    global backend, models, keras_utils
    backend, lyrs, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # New DNCON2 Inception-Resnet-V2 Architecture (2) #
    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=75,
        min_size=75,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid')
    x = conv2d_bn(x, 32, 3, padding='valid')
    x = conv2d_bn(x, 64, 3)
    x = layers.MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, padding='valid')
    x = layers.MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1)
    branch_1 = conv2d_bn(x, 48, 1)
    branch_1 = conv2d_bn(branch_1, 64, 5)
    branch_2 = conv2d_bn(x, 64, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3)
    branch_pool = layers.AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else 3
    x = layers.Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 256, 3)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_6a')(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv2d_bn(x, 256, 1)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv2d_bn(x, 256, 1)
    branch_2 = conv2d_bn(branch_2, 288, 3)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = layers.MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = layers.Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b')

    if include_top:
        # Classification block
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='inception_resnet_v2')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5'
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='e693bd0210a403b3192acc6073ad2e96')
        else:
            fname = ('inception_resnet_v2_weights_'
                     'tf_dim_ordering_tf_kernels_notop.h5')
            weights_path = keras_utils.get_file(
                fname,
                BASE_WEIGHT_URL + fname,
                cache_subdir='models',
                file_hash='d19885ff4a710c122648d3b5c3b684e4')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


# endregion


# region Training Code
def prepare_data(feature_folder, label_folder):
    import glob
    import os
    features = glob.glob(feature_folder + '/' + '*.txt')
    feature_files = []
    contact_files = []
    for feature in features:
        base = os.path.basename(feature)
        target = os.path.splitext(base)[0]

        if os.path.isfile(label_folder + '/' + target[2:] + '.dist'):
            feature_files.append(feature)
            contact_files.append(label_folder + '/' + target[2:] + '.dist')

    return feature_files, contact_files


def extract_features(fileX, L_MAX):
    x, seq_len = getX(fileX, L_MAX)
    F = len(x[0, 0, :])
    X_data = np.zeros((1, L_MAX, L_MAX, F))
    X_data[0, :, :, :] = x
    return X_data


def extract_contact(file_distance, L_MAX):
    distance = np.loadtxt(file_distance)
    contact = np.zeros((L_MAX, L_MAX), dtype=int)
    for i in range(distance.shape[0]):
        if distance[i, 2] > 8:
            contact[int(distance[i, 0]), int(distance[i, 1])] = 0
        else:
            contact[int(distance[i, 0]), int(distance[i, 1])] = 1
            contact[int(distance[i, 1]), int(distance[i, 0])] = 1
    return contact


def train_model(model_arch, file_weights, L_MAX, num_of_inputs_to_use):
    seq_len = 0
    generated_features = []
    generated_contacts = []
    feature_failed_to_generate = False
    current_working_dir = os.getcwd()
    X_train, X_val, X_test, y_train, y_val, y_test = \
        np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # Manually modify base_cached_data_dir if running script from the scripts directory
    current_working_dir = current_working_dir[:-8] if current_working_dir[-7:] == 'scripts' else current_working_dir

    # Determine if the data set has already been cached
    data_set_already_compiled = \
        os.path.exists(current_working_dir + '/databases/DNCON2/cached_features/X_train.npy') and \
        os.path.exists(current_working_dir + '/databases/DNCON2/cached_features/X_val.npy') and \
        os.path.exists(current_working_dir + '/databases/DNCON2/cached_features/X_test.npy') and \
        os.path.exists(current_working_dir + '/databases/DNCON2/cached_labels/y_train.npy') and \
        os.path.exists(current_working_dir + '/databases/DNCON2/cached_labels/y_val.npy') and \
        os.path.exists(current_working_dir + '/databases/DNCON2/cached_labels/y_test.npy')

    # Load cached data set into memory (if applicable)
    if data_set_already_compiled:
        X_train = np.load(current_working_dir + '/databases/DNCON2/cached_features/X_train.npy')
        X_val = np.load(current_working_dir + '/databases/DNCON2/cached_features/X_val.npy')
        X_test = np.load(current_working_dir + '/databases/DNCON2/cached_features/X_test.npy')
        y_train = np.load(current_working_dir + '/databases/DNCON2/cached_labels/y_train.npy')
        y_val = np.load(current_working_dir + '/databases/DNCON2/cached_labels/y_val.npy')
        y_test = np.load(current_working_dir + '/databases/DNCON2/cached_labels/y_test.npy')

    # Extracting features and labels from .txt file data set
    if not data_set_already_compiled:
        feature_files, contact_files = prepare_data(current_working_dir + '/databases/DNCON2/features',
                                                    current_working_dir + '/databases/DNCON2/labels')
        for i in range(num_of_inputs_to_use):
            try:
                extracted_feature = extract_features(feature_files[i], L_MAX)
                generated_features.append(extracted_feature[0, :, :, :])
            except IndexError:
                feature_failed_to_generate = True

            if not feature_failed_to_generate:
                try:
                    extracted_contact = extract_contact(contact_files[i], L_MAX)
                    generated_contacts.append(extracted_contact)
                except IndexError:
                    # Remove most recently added feature
                    del generated_features[-1]

            feature_failed_to_generate = False

        # Restructure features and labels
        features = np.array(generated_features)
        contacts = np.reshape(generated_contacts,
                              (num_of_inputs_to_use, len(generated_contacts) // num_of_inputs_to_use))

        # Construct a 60-20-20 (%) training-validation-testing data split
        X_train, X_test, y_train, y_test = train_test_split(features, contacts, test_size=0.2, random_state=1)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=1  # 0.25 x 0.8 = 0.2
        )

        X_train = np.asarray(X_train).astype(np.float32)
        X_val = np.asarray(X_val).astype(np.float32)
        X_test = np.asarray(X_test).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)
        y_val = np.asarray(y_val).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.float32)

    # Cache training-validation-testing data partitions
    if not data_set_already_compiled:
        np.save(current_working_dir + '/databases/DNCON2/cached_features/X_train', X_train)
        np.save(current_working_dir + '/databases/DNCON2/cached_features/X_val', X_val)
        np.save(current_working_dir + '/databases/DNCON2/cached_features/X_test', X_test)
        np.save(current_working_dir + '/databases/DNCON2/cached_labels/y_train', y_train)
        np.save(current_working_dir + '/databases/DNCON2/cached_labels/y_val', y_val)
        np.save(current_working_dir + '/databases/DNCON2/cached_labels/y_test', y_test)

    # Build model architecture
    input_shape = X_train[0].shape

    # Original DNCON2 architecture #
    model = build_orig_model_for_this_input_shape(model_arch, input_shape=input_shape)

    # Baseline Resnet architecture #
    # model = build_resnet_model_for_this_input_shape(input_shape=input_shape,
    #                                                 num_of_classes=2)

    # Inception-Resnet-V2 architecture #
    # model = build_inception_resnet_v2_model_for_this_input_shape(weights=file_weights, input_shape=input_shape,
    #                                                              backend=keras.backend, layers=keras.layers,
    #                                                              models=keras.models,
    #                                                              utils=keras.utils)

    # restore the model file "model.h5" from a specific run by user "lavanyashukla"
    # in project "save_and_restore" from run "10pr4joa"
    # best_model = wandb.restore('model-best.h5', run_path="amorehead/dncon2/1zsthi0f")

    # use the "name" attribute of the returned object if your framework expects a filename, e.g. as in Keras
    # model.load_weights(best_model.name)

    # Finalize model initialization
    model.compile(optimizer='Adam', loss='binary_crossentropy',
                  metrics=['acc', 'MeanSquaredError', 'AUC', 'Precision', 'Recall'])

    # Log metrics with wandb
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=128, epochs=500, callbacks=[WandbCallback()])

    # Evaluate model
    scores = model.evaluate(X_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # Save model to wandb
    model.save(os.path.join(wandb.run.dir, 'model-best.h5'))

# endregion
