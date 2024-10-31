import os
import pickle
import shutil
import zipfile
import logging
import datetime
import glob
import json
import time
import math

import larq as lq
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from PIL import Image
from prettytable import PrettyTable
import matplotlib.pyplot as plt


import tensorflow as tf
from larq import utils, math
from larq.quantizers import BaseQuantizer, _clipped_gradient, ste_sign
from collections import OrderedDict
import numpy as np

@utils.register_keras_custom_object
def k1_ste_activation(x):
    return math.sign(x)
         
def k_ste_sign(x, k_bit = 2, clip_value=1.0):

    #x = tf.clip_by_value(x, 0.0, clip_value)

    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value)

        zeros = tf.zeros_like(x)
         
        n = 2 ** (k_bit)
        n_h = 2 ** (k_bit-1)
         
        dY = 1/ (n-1)
         
        Y = OrderedDict()
        j = 0
        #for i in range(1,n_h,2):
        for i in range(0,n_h-1):
            Y[j] = (1 + 2*i) * dY * tf.math.sign(x)
            j += 1
            #print((1 + 2*i)*dY)
        Y[j] = 1.0 * tf.math.sign(x)
             
        dx = clip_value / n_h
        MASK = OrderedDict()
         
        MASK[0] = tf.math.less(tf.math.abs(x), 1*dx)
         
        for i in range(1, n_h-1):
            A = tf.math.greater(tf.math.abs(x), dx * (i))
            B = tf.math.greater(tf.math.abs(x), dx * (i+1))
            MASK[i] = tf.math.logical_xor(A, B)
            #print("{} - {}".format(dx*i, dx * (i+1)))
             
        MASK[n_h-1] = tf.math.greater(tf.math.abs(x), clip_value - dx)
             
        val = tf.where(MASK[0], Y[0], zeros)
        for j in range(1,len(Y)):
            val += tf.where(MASK[j], Y[j], zeros)
            
        
        return val, grad

    return _call(x)


@utils.register_keras_custom_object
def k_pos_ste(inputs, clip_value=1.0):
    
    #x = tf.clip_by_value(inputs, 0.0, 1.0)
    x = inputs
    
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value=clip_value)
        
        zeros = tf.zeros_like(x)
        val = tf.math.maximum(zeros, math.sign(x))
    
        return val, grad
        
    return _call(x) 
    
@utils.register_keras_custom_object
def k_round_ste(inputs, clip_value=1.0):
    
    x = tf.clip_by_value(inputs, 0.0, 1.0)
    #x = inputs
    
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value=clip_value)
        
        zeros = tf.zeros_like(x)
        val = tf.math.round(x)
    
        return val, grad
        
    return _call(x)

@utils.register_keras_custom_object
def k_scaled_round_ste(inputs, clip_value=1.0):
    
    x = tf.clip_by_value(inputs, 0.0, 1.0)
    #x = inputs
    
    @tf.custom_gradient
    def _call(x):
        def grad(dy):
            return _clipped_gradient(x, dy, clip_value=clip_value)
        
        #zeros = tf.zeros_like(x)
        val = tf.math.round(x) * 0.25
    
        return val, grad
        
    return _call(x)

@utils.register_keras_custom_object
class KRoundedSte(BaseQuantizer):
    
    precision = None
    
    def __init__(self, k_bit, scale = 1.0, clip_value = 1.0, clip_gradient = True, **kwargs):
        self.precision = k_bit
        self.scale = scale
        self.clip_value = clip_value
        self.clip_gradient = clip_gradient
        super().__init__(**kwargs)
   
    def call(self, inputs):
        x = tf.clip_by_value(inputs, 0.0, 1.0)

        @tf.custom_gradient
        def _call(x):
            def grad(dy):
                if self.clip_gradient:
                    return _clipped_gradient(x, dy, clip_value=self.clip_value)
                else:
                    return dy
        
            n = 2 ** self.precision - 1
            val = self.scale * tf.round(x * n) / n
    
            outputs = (val, grad)
            
            return outputs
    
        return super().call(_call(x))

    def get_config(self):
        config = super(KRoundedSte, self).get_config()
        d = {
            "k_bit": self.precision, 
            "scale" : self.scale, 
            "clip_gradient" : self.clip_gradient, 
            "clip_value" : self.clip_value
            }
        config.update(d)
        return config 
    

@utils.register_keras_custom_object
class KRoundedSte2(BaseQuantizer):
    
    precision = None
    
    def __init__(self, k_bit, scale = 1.0, clip_value = 1.0, clip_gradient = True, **kwargs):
        self.precision = k_bit
        self.scale = scale
        self.clip_value = clip_value
        self.clip_gradient = clip_gradient
        super().__init__(**kwargs)
   
    def call(self, inputs):
        #s = tf.sign(inputs)
        #x = tf.multiply(inputs, s)
        x = tf.clip_by_value(inputs, -1.0, 1.0)
        #x = inputs
        
        @tf.custom_gradient
        def _call(x):
            def grad(dy):
                if self.clip_gradient:
                    return _clipped_gradient(x, dy, clip_value=self.clip_value)
                else:
                    return dy
        
            n = 2 ** (self.precision) - 1
            #centering the quantizaton around the x-axis
            val = (tf.round(x * n) / n)
            #val = tf.multiply(s, val)
            
            #val = tf.sign(x) * val
            val = tf.clip_by_value(val, -1.0, 1.0)
    
            outputs = (val, grad)
            
            return outputs
    
        return super().call(_call(x))

    def get_config(self):
        
        config = super(KRoundedSte2, self).get_config()
        d = {
            "k_bit": self.precision, 
            "scale" : self.scale, 
            "clip_gradient" : self.clip_gradient, 
            "clip_value" : self.clip_value
            }
        config.update(d)
        return config  


def test(x):
    if x < 0:
        return x
    else:
        return 0

@utils.register_keras_custom_object
class KSteSign(BaseQuantizer):
    
    precision = None

    def __init__(self, k_bit : int, clip_value: float = 1.0, **kwargs):
        self.clip_value = clip_value
        self.precision = k_bit
        super().__init__(**kwargs)

    def call(self, inputs):
        if self.precision == 1:
            out = ste_sign(inputs, clip_value=self.clip_value)
        else:
            out = k_ste_sign(inputs, k_bit=self.precision, clip_value=self.clip_value)
        
        return super().call(out)
    

    def get_config(self):
        
        config = super(KSteSign, self).get_config()
        d = {
            "k_bit": self.precision, 
            "clip_value" : self.clip_value
            }
        config.update(d)
        return config
   

@utils.register_keras_custom_object
class MyDoReFaQuantizer(BaseQuantizer):
    
    precision = None

    def __init__(self, k_bit: int = 2, **kwargs):
        self.precision = k_bit
        super().__init__(**kwargs)

    def call(self, inputs):
        inputs = tf.clip_by_value(inputs, 0, 1.0)

        @tf.custom_gradient
        def _k_bit_with_identity_grad(x):
            def grad(dy):
                return _clipped_gradient(x, dy, clip_value=1.0)
            n = 2 ** self.precision - 1
            return (tf.round(x * n)) / n, grad

        outputs = _k_bit_with_identity_grad(inputs)
        return super().call(outputs)

    def get_config(self):
        config = super(MyDoReFaQuantizer, self).get_config()
        d = {
            "k_bit": self.precision, 
            }
        config.update(d)
        return config


@tf.custom_gradient
def scaled_gradient(x: tf.Tensor, scale: float = 1.0) -> tf.Tensor:
    def grad(dy):
        # We don't return a gradient for `scale` as it isn't trainable
        return (dy * scale, 0.0)

    return x, grad


@utils.register_alias("lsq")
@utils.register_keras_custom_object
class LSQ(tf.keras.layers.Layer):
    r"""Instantiates a serializable k_bit quantizer as in the LSQ paper.

    # Arguments
    k_bit: number of bits for the quantization.
    mode: either "signed" or "unsigned", reflects the activation quantization scheme to
        use. When using this for weights, use mode "weights" instead.
    metrics: An array of metrics to add to the layer. If `None` the metrics set in
        `larq.context.metrics_scope` are used. Currently only the `flip_ratio` metric is
        available.

    # Returns
    Quantization function

    # References
    - [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153)
    """
    precision = None

    def __init__(self, k_bit: int = 2, mode="unsigned", **kwargs):
        self.precision = k_bit
        self.mode = mode

        if mode == "unsigned":
            self.q_n = 0.00
            self.q_p = float(2 ** self.precision - 1)
        elif mode in ["signed", "weights"]:
            self.q_p = float(2 ** (self.precision - 1)) - 1

            # For signed, we can use the full signed range, e.g. [-2, 1]
            if mode == "signed":
                self.q_n = -float(2 ** (self.precision - 1))
            # For weights, we use a symmetric range, e.g. [-1, 1]
            else:
                self.q_n = -float(2 ** (self.precision - 1) - 1)

        else:
            raise ValueError(f"LSQ received unknown mode: {mode}")

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.s = self.add_weight(
            name="s",
            initializer="ones",
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )
        self._initialized = self.add_weight(
            name="initialized",
            initializer="zeros",
            dtype=tf.dtypes.bool,
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

        # Assuming that by num_features they mean all the individual pixels.
        # You can also try the number of feature maps instead.
        self.g = float(1.0 / np.sqrt(np.prod(input_shape[1:]) * self.q_p))

        super().build(input_shape)

    def call(self, inputs):
        # Calculate initial value for the scale using the first batch
        self.add_update(
            self.s.assign(
                tf.cond(
                    self._initialized,
                    lambda: self.s,  # If already initialized, just use current value
                    # Otherwise, use the value below as initialization
                    lambda: (2.0 * tf.reduce_mean(tf.math.abs(inputs)))
                    / tf.math.sqrt(self.q_p),
                )
            )
        )
        self.add_update(self._initialized.assign(True))
        s = scaled_gradient(self.s, self.g)
        rescaled_inputs = inputs / s
        clipped_inputs = tf.clip_by_value(rescaled_inputs, self.q_n, self.q_p)

        @tf.custom_gradient
        def _round_ste(x):
            return tf.round(x), lambda dy: dy

        return _round_ste(clipped_inputs) * s

    def get_config(self):
        return {**super().get_config(), "k_bit": self.precision, "mode": self.mode}   

# Function to plot history
def display_history(fn, metrics=None):
    def plot(metric, history):
        results = history[metric]
        epochs = range(1, len(results) + 1)
        plt.plot(epochs, results, "-x")
        try:
            val_results = history["val_" + metric]
            period = len(results) // len(val_results)
            val_epochs = [(i + 1) * period for i in range(len(val_results))]
            plt.plot(val_epochs, val_results, "-x")
            plt.legend(["Train", "Val"], loc="upper right")
        except KeyError:
            plt.legend(["Train"], loc="upper left")
        plt.title(f"Model {metric}")
        plt.ylabel(metric)
        plt.xlabel("Epoch")
        plt.show()

    history = pickle.load(fn)
    if metrics is None:
        metrics = [k for k in history.keys() if not k.startswith("val_")]
    for metric in metrics:
        plot(metric, history)

# Base class for checkpoints
class CheckPoint:
    def __init__(self, save_dir, create_if_not_exist=True):
        self.save_dir = save_dir
        if create_if_not_exist and not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def clear(self):
        shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)

    def save_history(self):
        pass

    def load_history(self):
        pass

    def callback(self):
        pass

# Single checkpoint class
class SingleCheckPoint(CheckPoint):
    def __init__(self, save_dir, model_name, filename="model", create_if_not_exist=True, save_weights_only=True, enable_monitoring=False, monitor_metric="val_accuracy"):
        super().__init__(save_dir, create_if_not_exist)
        self.save_model_dir = os.path.join(self.save_dir, model_name)
        self.checkpoint_fn = filename
        self.ckp = os.path.join(self.save_model_dir, filename + (".ckpt" if save_weights_only else ".h5"))
        self.hist = os.path.join(self.save_model_dir, filename + ".hst")
        self.zip = os.path.join(self.save_model_dir, filename + "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".zip")
        self.save_weights_only = save_weights_only
        self.monitoring = enable_monitoring
        self.monitor_metric = monitor_metric

    def save_history(self, history_obj):
        with open(self.hist, "wb") as f:
            pickle.dump(history_obj.history, f)

    def load_history(self):
        with open(self.hist, "rb") as f:
            return pickle.load(f)

    def callback(self, verbose=1):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.ckp,
            verbose=verbose,
            save_weights_only=self.save_weights_only,
            monitor=self.monitor_metric if self.monitoring else None,
            save_best_only=self.monitoring,
            mode="max",
        )

    def check_model_files(self):
        return len(glob.glob(self.ckp + "*")) > 0

    def backup(self):
        if self.check_model_files():
            files_to_backup = glob.glob(self.ckp + "*") + glob.glob(self.hist + "*") + glob.glob(os.path.join(self.save_model_dir, "checkpoint") + "*")
            with zipfile.ZipFile(self.zip, mode="w") as newZip:
                for f in files_to_backup:
                    newZip.write(f)
            return self.zip

    @property
    def filename(self):
        return self.ckp

# Abstract callback class
class AbstractCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.gen = None

    def set_val_gen(self, val_gen):
        pass

# Timer callback
class TimerCallback(AbstractCallback):
    def __init__(self):
        super().__init__()
        self.start = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start
        print(f"Epoch {epoch} ran in {duration:.2f} seconds.\n")

# Model wrapper class
class ModelWrapper:
    def __init__(self, model_name, working_dir, model_name_2="checkpoint", logger_lvl=logging.ERROR, l_rate=0.0001, verbose=True, save_weights_only=True, enable_monitoring=False, monitor_metric="val_accuracy", tensorboard=False, enable_history=False, clean_history=False, **kwargs):
        self.model_name = model_name
        self.working_dir = working_dir
        self.verbose = verbose
        self.lr = l_rate
        self.logger = self._setup_logger(level=logger_lvl)
        self.checkpoint = SingleCheckPoint(save_dir=self.working_dir, model_name=self.model_name, filename=model_name_2, create_if_not_exist=True, save_weights_only=save_weights_only, enable_monitoring=enable_monitoring, monitor_metric=monitor_metric)
        self.cb = [self.checkpoint.callback()]
        self.metrics = []
        self.friendly_names = None
        self._init_friendly_names()
        self.inner_layers = {}
        self.k_bit = 0

    def _setup_logger(self, level=logging.ERROR):
        FORMAT = "%(message)s"
        logging.basicConfig(format=FORMAT, level=logging.INFO)
        logger = logging.getLogger("Logger")
        logger.setLevel(level=level)
        return logger

    def _init_friendly_names(self):
        self.friendly_names = {}

    def _add_callbacks(self, new_cbs):
        if isinstance(new_cbs, list):
            self.cb.extend(new_cbs)
        else:
            self.cb.append(new_cbs)

    def _load_weights(self, fn=None):
        fn = fn or self.checkpoint.filename
        self.model.load_weights(fn)
        self.logger.info(f"=> Weights loaded from {fn}")

    def save_model(self, fn=None, save_format="tf"):
        fn = fn or os.path.join(self.checkpoint.save_model_dir, f"best_model.{save_format}")
        self.model.save(fn)



import config

import models.abstract
import larq as lq
import tensorflow as tf
from prettytable import PrettyTable
import tensorflow.keras as keras
from models.extra.Kste import KSteSign

class QuantizedHShallow(ModelWrapper):
    
    def __init__(self, 
                 model_name, 
                 working_dir, 
                 nClasses, 
                 fc_units = 512, 
                 filters = (96,256,256), 
                 enable_history = False, 
                 activation_precision : int = 1, 
                 kernel_precision : int = 1, 
                 model_name_2 = "checkpoint", 
                 logger_lvl = 
                 config.log_lvl, 
                 l_rate = 0.0001, 
                 optimizer = None,
                 loss = 'categorical_crossentropy',
                 **kwargs):
        self.filters = filters
        super(QuantizedHShallow, self).__init__(model_name, working_dir, model_name_2, logger_lvl = logger_lvl, enable_history = enable_history,**kwargs)    
        
        self.nClasses = nClasses
        self.l_rate = l_rate
        self.activation_precision = activation_precision
        self.kernel_precision = kernel_precision 
        self.units = fc_units
        self.optimizer = optimizer
        self.loss = loss
        
        self.model = self._setup_model(verbose = False) 
        

    '''
        Creates a Keras Model.
    '''
    
    def _setup_model(self, **kwargs):
        
        if "verbose" in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        
        #tf.debugging.set_log_device_placement(True)
        
        self.model_name = self.model_name if self.model_name is not None else 'unknown_model_name'
        
        input_img = keras.layers.Input(shape = (227, 227, 3))
        cnn = self._cnn(input_tensor = input_img)
            
        net = self._fully_connected(self.nClasses, cnn,
                        units = self.units)
    
        model = keras.models.Model(input_img, net)
            
        if self.optimizer is None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.l_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name='Adam'
                )
            self.optimizer = optimizer
    
        model.compile(optimizer=self.optimizer,
                  loss=self.loss,
                  metrics=['accuracy'])
        
        self.model = model

        if verbose:
            self._display_layers()
        
        return self.model
    
    '''
        Returns the CNN part of the Model
    '''
    def _cnn(self,input_tensor=None, input_shape=None,
                use_bias = False,
                momentum = 0.9):
    

        img_input = keras.layers.Input(shape=input_shape) if input_tensor is None else (
            keras.layers.Input(tensor=input_tensor, shape=input_shape) if not keras.backend.is_keras_tensor(input_tensor) else input_tensor
        )
        
        #Block 1 - The input quantizer have to be set on None
        x = self._conv_block(filters = self.filters[0], kernel_size = (11,11), strides = (4,4), padding = 'valid', name='conv1',
                                            input_quantizer = None,
                                            kernel_quantizer = "ste_sign",
                                            kernel_constraint = "weight_clip",
                                            use_bias = use_bias,
                                            batch_norm = False,
                                            momentum = momentum
                                            )(img_input)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)            
        #Block 2
        x = self._conv_block(filters = self.filters[1], kernel_size = (5,5), strides = (1,1),  padding = 'same', name='conv2',
                                            input_quantizer = "ste_sign",
                                            kernel_quantizer = "ste_sign",
                                            kernel_constraint = "weight_clip",
                                            use_bias = use_bias,
                                            momentum = momentum)(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)  
        
        #Block 3
        x = self._conv_block(self.filters[2],  kernel_size = (3,3), strides = (1,1), name='conv3',
                                            input_quantizer = "ste_sign",
                                            kernel_quantizer = "ste_sign",
                                            kernel_constraint = "weight_clip",
                                            use_bias = use_bias,
                                            momentum = momentum)(x)
                                    
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)   
              
        x = keras.layers.BatchNormalization(name = 'pool5_bn', momentum = momentum)(x) 
         
        x = keras.layers.Flatten(name='flatten')(x)
        #self._latent_space = x
        
        return x
     
     
    '''
        Returns the FC part of the Model
    ''' 
    def _fully_connected(self, nClasses, cnn, units):
        
        x = self._dense_block(units = units, activation='relu', name='fc6')(cnn)
        x = self._dense_block(units = units, activation='relu', name='fc7')(x)
        x = self._dense_block(units = nClasses, activation='softmax', name='fc8')(x)
        
        return x

    '''
        Convolutional block
    '''
    def _conv_block(self, 
                    filters, 
                    name,
                    kernel_size = (3,3), 
                    strides = (1,1), 
                    padding = 'same',
                    pad_value = 1.0,
                    #activation=None, name = None,
                    input_quantizer = 'ste_sign',
                    kernel_quantizer = 'ste_sign',
                    kernel_constraint=lq.constraints.WeightClip(clip_value=1),
                    use_bias = False,
                    batch_norm = True,
                    momentum = 0.9):
        
        def layer_wrapper(inp):
            x = inp
            if batch_norm:
                x = keras.layers.BatchNormalization(name = name + '_bn', momentum = momentum)(x)
            x = lq.layers.QuantConv2D(filters, kernel_size = kernel_size, strides = strides, padding=padding, pad_values = pad_value, name=name,
                                      input_quantizer = input_quantizer,
                                      kernel_quantizer = kernel_quantizer,
                                      kernel_constraint = kernel_constraint,
                                      use_bias = use_bias)(x)
            
            #x = keras.layers.Activation(activation, name = name + '_act')(x)
            return x

        return layer_wrapper

    
    '''
        Dense block
    '''
    def _dense_block(self, units, activation='relu', name='fc1', use_batch_norm = True):

        def layer_wrapper(inp):
            x = lq.layers.QuantDense(units, name=name)(inp)
            if use_batch_norm:
                x = keras.layers.BatchNormalization(name='bn_{}'.format(name))(x)
            x = keras.layers.Activation(activation, name='act_{}'.format(name))(x)
            #x = keras.layers.Dropout(dropout, name='dropout_{}'.format(name))(x)
            return x

        return layer_wrapper  
        

    '''
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_name: the layer name as defined in the model
    '''
        
    def get_inner_layer_by_name(self, layer_name, k_bit = None, activation = None, flatten = False):
        layer = self.layer_output_by_name(layer_name)
        # Quantized depends on the network. For LarqAlex is ste_sign
        if activation:
            out = keras.layers.Activation(activation)(layer.output)
        else:
            out = layer.output
        if k_bit:
            out = KSteSign(k_bit=k_bit)(out)
        if flatten:
            out = keras.layers.Flatten()(out)
        model = tf.keras.models.Model(inputs=self.model.input, outputs = out)
    
        return model
    
    def _display_layers(self):
        c = 0
        t = PrettyTable(['#','Layer','in','out','Trainable'])
        for l in self.model.layers: 
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)  



# Example usage
model = QuantizedHShallow(
    model_name="floppynet_TRO",
    working_dir="",
    model_name_2="model",
    logger_lvl="WARNING",
    nClasses=365,
    fc_units=256,
    filters=(96, 256, 256),
    l_rate=5e-4,
    save_weights_only=True,
    enable_monitoring=True,
    tensorboard=True,
    activation_precision=1,
    kernel_precision=1,
    enable_history=False,
    clean_history=True,
    optimizer=None,
    loss="categorical_crossentropy",
)
