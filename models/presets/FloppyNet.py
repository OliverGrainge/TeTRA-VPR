import datetime
import glob
import json
import logging
import os
import pickle
import shutil
import time
import zipfile
from datetime import datetime

import larq as lq
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import torch
from PIL import Image
from prettytable import PrettyTable
from tensorflow.keras.callbacks import TensorBoard
from utils.results import display_history

import config
import models.abstract
import utils
from models.extra.Kste import KSteSign
from utils import clear_dir, mkdirs

"""
    Base class to handle checkpoint filenames
"""


class CheckPoint:
    """
    Constructor
    #Arguments
        save_dir: is thepath were the model files will be stored
        create_is_not_exists: if true makes the save_dir directory being create by the class  constructor
    """

    def __init__(self, save_dir, create_if_not_exist=True):
        self.save_dir = save_dir

        if create_if_not_exist == True:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    """
        Clear the model directory.
    """

    def clear(self):
        shutil.rmtree(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    """
        Save the history file along with the model
    """

    def save_history(self):
        pass

    """
        Restore history
    """

    def load_history(self):
        pass

    """
        Callback to pass to model.fit
    """

    def callback(self):
        pass


"""
    Derived class of CheckPoints. Handles the filename for the single save mode.
"""


class SingleCheckPoint(CheckPoint):
    """
    Constructor
    #Arguments
        save_dir: is thepath were the model files will be stored
        model_name: is the name of the file to save. For example: model_xyz will saved as model_xyz.ckpt in save_dir
        create_is_not_exists: if true makes the save_dir directory being create by the class  constructor
    """

    def __init__(
        self,
        save_dir,
        model_name,
        filename="model",
        create_if_not_exist=True,
        save_weights_only=True,
        enable_monitoring=False,
        monitor_metric="val_accuracy",
    ):

        CheckPoint.__init__(self, save_dir, create_if_not_exist=create_if_not_exist)

        self.save_model_dir = os.path.join(self.save_dir, model_name)

        self.checkpoint_fn = filename

        if not os.path.exists(self.save_model_dir) and create_if_not_exist == True:
            os.makedirs(self.save_model_dir)

        self.ckp = os.path.join(self.save_model_dir, filename + ".ckpt")
        self.hist = os.path.join(self.save_model_dir, filename + ".hst")
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        self.zip = os.path.join(
            self.save_model_dir, filename + "_" + date_time + ".zip"
        )

        self.save_weights_only = save_weights_only
        if save_weights_only:
            self.ckp = os.path.join(self.save_model_dir, filename + ".ckpt")
        else:
            self.ckp = os.path.join(self.save_model_dir, filename + ".h5")

        self.monitoring = enable_monitoring
        self.monior_metric = monitor_metric

    """
        Save the history along with the model
    """

    def save_history(self, history_obj):

        with open(self.hist, "wb") as f:
            pickle.dump(history_obj.history, f)

    """
        The fullpath of the model file
    """

    def load_history(self):
        with open(self.hist, "rb") as f:
            history = pickle.load(f)
            return history

    """
        Callback to pass to model.fit
    """

    def callback(self, save_weights_only=True, verbose=1):

        if self.monitoring:
            fn = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.ckp,
                verbose=verbose,
                save_weights_only=self.save_weights_only,
                monitor=self.monior_metric,
                save_best_only=True,
                mode="max",
            )
        else:

            fn = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.ckp,
                verbose=verbose,
                save_weights_only=self.save_weights_only,
            )

        return fn

    """
        Checks if a model file already exists
    """

    def check_model_files(self):
        ckp = glob.glob(self.ckp + "*")
        print(len(ckp))
        hst = glob.glob(self.hist + "*")
        print(len(hst))
        return len(ckp) > 0

    """ 
        Backups older files
    
    """

    def backup(self):
        if self.check_model_files():
            files_to_backup = []
            for f in list(glob.glob(self.ckp + "*")):
                files_to_backup.append(f)
            for f in list(glob.glob(self.hist + "*")):
                files_to_backup.append(f)
            for f in list(
                glob.glob(os.path.join(self.save_model_dir, "checkpoint") + "*")
            ):
                files_to_backup.append(f)
            # print(file_to_backup)

            with zipfile.ZipFile(self.zip, mode="w") as newZip:
                for f in files_to_backup:
                    newZip.write(f)
            return self.zip

    @property
    def filename(self):
        return self.ckp


"""
    This checkpoint saver has been added lately to the framework to handle binary networks.
    Binary Network exhibits wide variation in validation accuracy, thus just saving the last model or using monitoring, is not always the
    the best way to spot the best model for VPR.
    This callback saves all the checkpoints in weights only format and, optionally, ad a complete model.
    
"""


class MultiBrenchCheckPointSaver(CheckPoint):
    """
    @param save_mode: 'weights' for weights only, 'complete' for complete model.
    """

    def __init__(
        self,
        save_dir,
        model_name,
        save_mode,
        filename="model_{epoch:03d}_{val_accuracy:.2f}",
        create_if_not_exist=True,
        clean=True,
    ):
        CheckPoint.__init__(self, save_dir, create_if_not_exist=create_if_not_exist)

        self.save_model_dir_weights = os.path.join(
            self.save_dir, model_name, "history", "weights"
        )
        self.save_model_dir_complete = os.path.join(
            self.save_dir, model_name, "history", "complete"
        )

        self.save_mode = save_mode

        if clean:
            shutil.rmtree(self.save_model_dir_weights, ignore_errors=True)
            shutil.rmtree(self.save_model_dir_complete, ignore_errors=True)

        if self.save_mode == "weights":
            if (
                not os.path.exists(self.save_model_dir_weights)
                and create_if_not_exist == True
            ):
                os.makedirs(self.save_model_dir_weights)

        if self.save_mode == "complete":
            if (
                not os.path.exists(self.save_model_dir_complete)
                and create_if_not_exist == True
            ):
                os.makedirs(self.save_model_dir_complete)

        self.ckp_w = os.path.join(self.save_model_dir_weights, filename + ".ckpt")
        self.ckp_c = os.path.join(self.save_model_dir_complete, filename + ".h5")

    def callback(self):

        if self.save_mode == "weights":
            fn = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.ckp_w, verbose=True, save_weights_only=True
            )

        if self.save_mode == "complete":
            fn = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.ckp_c, verbose=True, save_weights_only=False
            )

        return fn


"""
    This Abstract Class has the purpose of providing my custom callbacks
    with the method set_val_gen, which is required for particular cases such as the autoencoders.
"""


class AbstractCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        super(AbstractCallback, self).__init__()
        self.gen = None

    def set_val_gen(self, val_gen):
        pass


class TimerCallback(AbstractCallback):

    def __init__(self):
        super(TimerCallback, self).__init__()
        self.start = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.start
        print("Epoch {:d} ran in {:0.2f} seconds.\n".format(epoch, duration))


class SampleAutoencoderReconstraction(AbstractCallback):

    def __init__(self, out_dir, generator, create_dir=True, clear_old=False):
        super(SampleAutoencoderReconstraction, self).__init__()
        self.img_index = 0
        self.dir = out_dir
        self.gen = generator

        if clear_old:
            clear_dir(out_dir)

        if create_dir:
            mkdirs(out_dir)

    def set_val_gen(self, val_gen):
        AbstractCallback.set_val_gen(self, val_gen)
        self.gen = val_gen

    def on_epoch_end(self, epoch, logs={}):
        batch = next(self.gen)
        y_pred = self.model.predict(batch)

        b_img = (batch[0][0][:, :, :] * 255).astype(np.uint8)

        fn = os.path.join(self.dir, "{:04d}_sample.png".format(epoch))
        r_img = (y_pred[0][:, :, :] * 255).astype(np.uint8)

        img = Image.fromarray(np.concatenate((b_img, r_img), axis=1))
        img.save(fn)

        print("\nSample save at {:s}".format(fn))


"""
    Defines a set of general purpose operations to execute at the end of various training phases
"""


class MiscellaneousCallback(AbstractCallback):

    def __init__(self, out_dir):
        super(MiscellaneousCallback, self).__init__()
        self.dir = out_dir
        self.epoch_fn = "epoch.ckpt"

    def on_epoch_end(self, epoch, logs={}):
        self._annotate_epoch(epoch)

    def _annotate_epoch(self, epoch):
        fn = os.path.join(self.dir, self.epoch_fn)
        with open(fn, "w") as f:
            x = {"last_epoch_trained": epoch}
            json.dump(x, f)

    def read_epoch_annotation(self):
        fn = os.path.join(self.dir, self.epoch_fn)
        if os.path.exists(fn):
            with open(fn, "r") as f:
                x = json.load(f)
                return x["last_epoch_trained"]
        else:
            return 0


class ClassifierTensorboardCallback(tf.keras.callbacks.TensorBoard):

    def __init__(
        self,
        log_dir="logs",
        histogram_freq=0,
        write_graph=False,
        write_images=False,
        update_freq="epoch",
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None,
        **kwargs
    ):
        super(ClassifierTensorboardCallback, self).__init__(
            log_dir,
            histogram_freq,
            write_graph,
            write_images,
            update_freq,
            profile_batch,
            embeddings_freq,
            embeddings_metadata,
        )


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir="./logs", **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, "training")
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, "validation")

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.compat.v1.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {
            k.replace("val_", ""): v for k, v in logs.items() if k.startswith("val_")
        }
        for name, value in val_logs.items():
            summary = tf.compat.v1.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class LRTensorBoard(TensorBoard):
    def __init__(
        self, log_dir, **kwargs
    ):  # add other arguments to __init__ if you need
        super().__init__(log_dir=log_dir, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # logs.update({'lr': tf.keras.backend.eval(self.model.optimizer.lr)})
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)


class ModelWrapper:
    """
    Contructor
        @param model_name: the name of the model is part will of the fullpath of the complete working dir
        @Param working_dir: the root of the working folder where models, output, etc.. will be saved
        @param model_name2: is the name for the weight file
        @param logger_lvl: loggin level for the internal logger
        @param l_rate: learning rate
        @param **kwargs: is deputed to handle the parameter for the derived classes
    """

    def __init__(
        self,
        model_name,
        working_dir,
        model_name_2="checkpoint",
        logger_lvl=logging.ERROR,
        l_rate=0.0001,
        verbose=True,
        save_weights_only=True,
        enable_monitoring=False,
        monitor_metric="val_accuracy",
        tensorboard=False,
        enable_history=False,
        clean_history=False,
        **kwargs
    ):
        self.model_name = model_name
        self.working_dir = working_dir
        self.verbose = verbose

        self.lr = l_rate

        self.logger = self._setup_logger(level=logger_lvl)

        self.checkpoint = SingleCheckPoint(
            save_dir=self.working_dir,
            model_name=self.model_name,
            filename=model_name_2,
            create_if_not_exist=True,
            save_weights_only=save_weights_only,
            enable_monitoring=enable_monitoring,
            monitor_metric=monitor_metric,
        )

        # Add the checkpoint callback to the callbacks to call during the training
        self.cb = [
            self.checkpoint.callback(),
        ]

        if enable_history:
            self.history_saver = MultiBrenchCheckPointSaver(
                save_dir=self.working_dir,
                # this is to have 'history' as a subdirectory of model subfolder
                model_name=self.model_name,
                save_mode="weights",
                clean=clean_history,
            )
            self._add_callbacks(self.history_saver.callback())

        if not save_weights_only:
            # Experimental
            self.checkpoint2 = SingleCheckPoint(
                save_dir=self.working_dir,
                model_name=self.model_name,
                filename=model_name_2,
                create_if_not_exist=True,
                save_weights_only=True,
                enable_monitoring=enable_monitoring,
                monitor_metric=monitor_metric,
            )
            self._add_callbacks(self.checkpoint2.callback())

        self.save_weights_only = save_weights_only

        if tensorboard:
            tb_dir_scalars = os.path.join(
                self.working_dir,
                self.model_name,
                r"logs\\scalars\\" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            )  # @UndefinedVariable
            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=tb_dir_scalars, write_graph=True
            )
            # tensorboard_callback = TrainValTensorBoard(log_dir=tb_dir_scalars)
            #### self._add_callbacks(tensorboard_callback)
            # NEW (24.11.2020)
            lr_log_callback = LRTensorBoard(
                log_dir=tb_dir_scalars, write_graph=True, profile_batch=0
            )
            self._add_callbacks(lr_log_callback)

        # Metrics
        self.metrics = []

        # Create the model. Later in the process, weights might be loaded
        # Let's try to commenti it out
        # self.model = self._setup_model(**kwargs)

        # layer map. Needs to be implemented in the subclass
        self.friendly_names = None
        self._init_friendly_names()

        self.inner_layers = {}

        self.k_bit = 0

    def input_shape(self):
        input_shape = self.model._feed_input_shapes[0]
        IMG_HEIGHT = input_shape[1]
        IMG_WIDTH = input_shape[2]
        return (IMG_HEIGHT, IMG_WIDTH)

    def output_shape(self):
        return self.model.layers[-1].output.shape

    """
        Trains the model using Keras generators
        
        NOTE: working folder == self.checkpoint.save_model_dir, which is a subfolder of self.working_dir
        
            - epochs: training epochs
            - train_gen: training generator
            - val_gen; validation generator
            - resume_training: set True to resume from a previous checkpoint
            - weight_filename: used only when 'resume_training' is True. 
                If it is 'None' the last checkpoint in the working area is loaded, otherwise the specified file is used
            - BACKUP_OLD_MODEL_DATA: if True a backup of the old weights file is created in the working folder.
    """

    def fit_generator(
        self,
        epochs,
        train_gen,
        val_gen,
        resume_training=False,
        weight_filename=None,
        BACKUP_OLD_MODEL_DATA=False,
    ):

        self.fit(
            epochs,
            train_gen,
            val_gen,
            resume_training,
            weight_filename,
            BACKUP_OLD_MODEL_DATA,
        )

    def fit(
        self,
        epochs,
        train_gen,
        val_gen,
        resume_training=False,
        weight_filename=None,
        BACKUP_OLD_MODEL_DATA=False,
    ):

        # Set the val_generators to the callbacks which needs it
        # Skip the first as it is a function and not a Callbackobject

        for i in range(0, len(self.cb)):
            try:
                self.cb[i].set_val_gen(val_gen)
            except:
                pass

        logger = self.logger

        logger.info("=> Checkpoint set at: {0}\n".format(self.checkpoint.filename))

        # Backup old model weights if required
        if BACKUP_OLD_MODEL_DATA:
            bkp_zip_fn = self.checkpoint.backup()
            if not bkp_zip_fn is None:
                logger.info("=> Old data saved in {:s}".format(bkp_zip_fn))

        # Epoch Annotation callback
        epoch_tracker = MiscellaneousCallback(self.checkpoint.save_model_dir)
        checkpoint_fn = os.path.join(self.checkpoint.save_model_dir, "checkpoint")
        if resume_training and os.path.exists(checkpoint_fn):
            initial_epoch = epoch_tracker.read_epoch_annotation()
            # Load weights to resume training
            self._load_weights(fn=self.checkpoint.filename)
            #             try:
            #                 self._load_model(fn = os.path.join(self.checkpoint.filename))
            #             except:
            #                 print("Load Model Failed. Trying with weights only")
            #                 self._load_weights(fn = self.checkpoint.filename)
            msg = utils.assign(weight_filename, self.checkpoint.filename)
            logger.info("=> Weights file {:s}\n".format(msg))
            logger.info(
                "=> Training will resume from epoch {0}\n".format(initial_epoch)
            )
        else:
            initial_epoch = 0
            logger.info("=> Training will start from the beginning\n")

        self._add_callbacks(
            [
                epoch_tracker,
            ]
        )

        # Time measure
        tcb = TimerCallback()
        self._add_callbacks(
            [
                tcb,
            ]
        )

        STEP_SIZE_TRAIN = train_gen.n // train_gen.batch_size
        if not val_gen is None:
            STEP_SIZE_VALID = val_gen.n // val_gen.batch_size
        else:
            STEP_SIZE_VALID = 1

        logger.info("=> Start training...\n")

        # I need to check the version in order to call the proper method
        if self._tf_version_230():

            history = self.model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                steps_per_epoch=STEP_SIZE_TRAIN,
                validation_steps=STEP_SIZE_VALID,
                callbacks=self.cb,
                workers=2,
                validation_freq=1,
                initial_epoch=initial_epoch,
                # metrics = self.metrics,
            )
        else:
            history = self.model.fit_generator(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                steps_per_epoch=STEP_SIZE_TRAIN,
                validation_steps=STEP_SIZE_VALID,
                callbacks=self.cb,
                workers=2,
                validation_freq=1,
                initial_epoch=initial_epoch,
                # metrics = self.metrics,
            )

        self.checkpoint.save_history(history)

    @property
    def save_model_dir(self):
        return self.checkpoint.save_model_dir

    def eval(self):
        pass

    def predict(self):
        pass

    """
        Returns a model cut at the required level
        Deprecated: use layer_output_by_name instead
    """

    def layer_output(self, layer_friendly_name, k_bit=0):
        layer = self.friendly_names[layer_friendly_name]
        return self._layer_output(layer, k_bit)

    """
        Returns a model cut at the required level
        @params
            - name: layer name as defined in the model
    """

    def layer_output_by_name(self, name, k_bit=0):
        layer = 0
        for l in self.model.layers:
            if l.name == name:
                return self._layer_output(layer, k_bit)
            else:
                layer += 1

    def _layer_output(self, layer, k_bit=0):

        l = self.model.layers[layer]
        out = l.output

        if k_bit > 0:
            out = lq.quantizers.DoReFaQuantizer(k_bit=k_bit)(out)
        out = tf.keras.models.Model(inputs=self.model.input, outputs=out)

        return out

    def layer_index_by_name(self, name):
        layer = 0
        for l in self.model.layers:
            if l.name == name:
                return layer
            else:
                layer += 1

    """
        Interface to add an intermediate layer to the available model to extract inner features
        layer_friendly: the label of the layer.
        layer_model: the keras model built from the inner layer.
        override: if true, the layer model is replaced (if already exists).
    """

    def _add_inner_layer(self, layer_friendly, layer_model, override=True):

        if override or not layer_friendly in self.inner_layers:
            self.inner_layers[layer_friendly] = layer_model
            return True
        else:
            return False

    """
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_frindly: the label identifying the layer to get
        Deprecated: use get_inner_layer_by_name instead
    """

    def get_inner_layer(self, layer_friendly):
        if layer_friendly in self.inner_layers:
            return self.inner_layers[layer_friendly]
        else:
            layer_model = self.layer_output(layer_friendly, k_bit=self.k_bit)
            self._add_inner_layer(layer_friendly, layer_model)
            return layer_model

    """
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_name: the layer name as defined in the model
    """

    def get_inner_layer_by_name(self, layer_name, **kwargs):
        if layer_name in self.inner_layers:
            return self.inner_layers[layer_name]
        else:
            layer_model = self.layer_output_by_name(layer_name, k_bit=self.k_bit)
            self._add_inner_layer(layer_name, layer_model)
            return layer_model

    """
        Initializes the layer index mapping 
    """

    def _init_friendly_names(self):
        self.friendly_names = {}

    def _set_metrics(self, metrics):
        self.metrics = metrics

    """
        Add callbacks to use during the training phase
    """

    def _add_callbacks(self, new_cbs):
        try:
            for cb in new_cbs:
                self.cb.append(cb)
        except:
            self.cb.append(new_cbs)

    """
        Creates the Keras Model. This method should also compile the model with the proper optimizer, metrics and loss
    """

    def _setup_model(self, **kwargs):
        pass

    """
        Load weights for the model.
            - fn: the checkpoint filename. If fn is 'None', the internal checkpoint.filename will be used to load the latest model.
    """

    def _load_weights(self, fn=None):
        if fn is None:
            fn = self.checkpoint.filename
        else:
            fn = fn
        # if os.path.exists(fn):
        self.model.load_weights(fn)
        self.logger.info("=> Weights loaded from {:s}".format(fn))

    def _load_model(self, fn=None):

        if fn is None:
            fn = self.checkpoint.filename
        else:
            fn = fn
        # if os.path.exists(fn):
        # tf.keras.backend.clear_session()
        self.model = tf.keras.models.load_model(fn)
        self.logger.info("=> Model loaded from {:s}".format(fn))

    """
        fit_generator is deprecated since 2.3.0. I need to check the version in order to call the proper method
    """

    def _tf_version_230(self):
        v = tf.__version__
        t = v.split(".")
        return int(t[0]) >= 2 and int(t[1]) >= 3

    """
        Saves full model into a directory
        @param format: can be either 'tf' or 'h5' 
    """

    def save_model(self, fn=None, save_format="tf"):
        if fn is None:
            fn = os.path.join(
                self.checkpoint.save_model_dir, "best_model." + save_format
            )
        else:
            fn = fn
        self.model.save(fn)

    """
        Wrapper for self._load_weights
        Load weights for the model.
            - fn: the checkpoint filename. If fn is 'None', the internal checkpoint.filename will be used to load the latest model.
    """

    def load_weights(self, fn=None):
        self._load_weights(fn=fn)

    def load(self, fn=None):
        if self.save_weights_only:
            self._load_weights(fn=fn)
        else:
            self._load_model(fn=fn)

    """
        Configures the logger
    """

    def _setup_logger(self, level=logging.ERROR):
        # Logging setup
        FORMAT = "%(message)s"
        logging.basicConfig(format=FORMAT, level=logging.INFO)
        logger = logging.getLogger("Logger")
        logger.setLevel(level=level)

        return logger

    """
        Plots the metrics set for the model
    """

    def display_history(self):
        fn = self.checkpoint.hist
        if os.path.exists(fn):
            display_history(fn=fn)
        else:
            self.logger.info("=> History file does not exists: {:s}".format(fn))

    """
        This is a single conv block I frequently use . 
        Override is as convenient for your network.
        note:  CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
    """

    def _conv_block(
        self, units, kernel_size=(3, 3), activation="relu", block=1, layer=1
    ):

        def layer_wrapper(inp):
            x = lq.layers.QuantConv2D(
                units,
                activation=None,
                kernel_size=kernel_size,
                padding="same",
                name="block{}_conv{}".format(block, layer),
            )(inp)
            x = keras.layers.BatchNormalization(
                name="block{}_bn{}".format(block, layer)
            )(x)
            x = keras.layers.Activation(
                activation, name="block{}_act{}".format(block, layer)
            )(x)
            return x

        return layer_wrapper

    """
        This is dense block  I frequently use . 
        Override is as convenient for your network.
        note:  CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
    """

    def _dense_block(self, units, activation="relu", name="fc1"):

        def layer_wrapper(inp):
            x = lq.layers.QuantDense(units, name=name, activation=None)(inp)
            x = keras.layers.BatchNormalization(name="{}_bn".format(name))(x)
            x = keras.layers.Activation(activation, name="{}_act".format(name))(x)
            return x

        return layer_wrapper


"""
    With respect to ModelWrapper adds methods that are apecific for an autoencoder
"""


class QuantizedHShallow(ModelWrapper):

    def __init__(
        self,
        model_name,
        working_dir,
        nClasses,
        fc_units=512,
        filters=(96, 256, 256),
        enable_history=False,
        activation_precision: int = 1,
        kernel_precision: int = 1,
        model_name_2="checkpoint",
        logger_lvl=config.log_lvl,
        l_rate=0.0001,
        optimizer=None,
        loss="categorical_crossentropy",
        **kwargs
    ):
        self.filters = filters
        super(QuantizedHShallow, self).__init__(
            model_name,
            working_dir,
            model_name_2,
            logger_lvl=logger_lvl,
            enable_history=enable_history,
            **kwargs
        )

        self.nClasses = nClasses
        self.l_rate = l_rate
        self.activation_precision = activation_precision
        self.kernel_precision = kernel_precision
        self.units = fc_units
        self.optimizer = optimizer
        self.loss = loss

        self.model = self._setup_model(verbose=False)

    """
        Creates a Keras Model.
    """

    def _setup_model(self, **kwargs):

        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
        else:
            verbose = False

        # tf.debugging.set_log_device_placement(True)

        self.model_name = (
            self.model_name if self.model_name is not None else "unknown_model_name"
        )

        input_img = keras.layers.Input(shape=(227, 227, 3))
        cnn = self._cnn(input_tensor=input_img)

        net = self._fully_connected(self.nClasses, cnn, units=self.units)

        model = keras.models.Model(input_img, net)

        if self.optimizer is None:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.l_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
                name="Adam",
            )
            self.optimizer = optimizer

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=["accuracy"])

        self.model = model

        if verbose:
            self._display_layers()

        return self.model

    """
        Returns the CNN part of the Model
    """

    def _cnn(self, input_tensor=None, input_shape=None, use_bias=False, momentum=0.9):

        img_input = (
            keras.layers.Input(shape=input_shape)
            if input_tensor is None
            else (
                keras.layers.Input(tensor=input_tensor, shape=input_shape)
                if not keras.backend.is_keras_tensor(input_tensor)
                else input_tensor
            )
        )

        # Block 1 - The input quantizer have to be set on None
        x = self._conv_block(
            filters=self.filters[0],
            kernel_size=(11, 11),
            strides=(4, 4),
            padding="valid",
            name="conv1",
            input_quantizer=None,
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=use_bias,
            batch_norm=False,
            momentum=momentum,
        )(img_input)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool1")(x)
        # Block 2
        x = self._conv_block(
            filters=self.filters[1],
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            name="conv2",
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=use_bias,
            momentum=momentum,
        )(x)
        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool2")(x)

        # Block 3
        x = self._conv_block(
            self.filters[2],
            kernel_size=(3, 3),
            strides=(1, 1),
            name="conv3",
            input_quantizer="ste_sign",
            kernel_quantizer="ste_sign",
            kernel_constraint="weight_clip",
            use_bias=use_bias,
            momentum=momentum,
        )(x)

        x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name="pool5")(x)

        x = keras.layers.BatchNormalization(name="pool5_bn", momentum=momentum)(x)

        x = keras.layers.Flatten(name="flatten")(x)
        # self._latent_space = x

        return x

    """
        Returns the FC part of the Model
    """

    def _fully_connected(self, nClasses, cnn, units):

        x = self._dense_block(units=units, activation="relu", name="fc6")(cnn)
        x = self._dense_block(units=units, activation="relu", name="fc7")(x)
        x = self._dense_block(units=nClasses, activation="softmax", name="fc8")(x)

        return x

    """
        Convolutional block
    """

    def _conv_block(
        self,
        filters,
        name,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        pad_value=1.0,
        # activation=None, name = None,
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint=lq.constraints.WeightClip(clip_value=1),
        use_bias=False,
        batch_norm=True,
        momentum=0.9,
    ):

        def layer_wrapper(inp):
            x = inp
            if batch_norm:
                x = keras.layers.BatchNormalization(
                    name=name + "_bn", momentum=momentum
                )(x)
            x = lq.layers.QuantConv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                pad_values=pad_value,
                name=name,
                input_quantizer=input_quantizer,
                kernel_quantizer=kernel_quantizer,
                kernel_constraint=kernel_constraint,
                use_bias=use_bias,
            )(x)

            # x = keras.layers.Activation(activation, name = name + '_act')(x)
            return x

        return layer_wrapper

    """
        Dense block
    """

    def _dense_block(self, units, activation="relu", name="fc1", use_batch_norm=True):

        def layer_wrapper(inp):
            x = lq.layers.QuantDense(units, name=name)(inp)
            if use_batch_norm:
                x = keras.layers.BatchNormalization(name="bn_{}".format(name))(x)
            x = keras.layers.Activation(activation, name="act_{}".format(name))(x)
            # x = keras.layers.Dropout(dropout, name='dropout_{}'.format(name))(x)
            return x

        return layer_wrapper

    """
        Returns a runnable model to extract inner feature. If it not exists, then it is instantiated and added to self.inner_layers
        layer_name: the layer name as defined in the model
    """

    def get_inner_layer_by_name(
        self, layer_name, k_bit=None, activation=None, flatten=False
    ):
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
        model = tf.keras.models.Model(inputs=self.model.input, outputs=out)

        return model

    def _display_layers(self):
        c = 0
        t = PrettyTable(["#", "Layer", "in", "out", "Trainable"])
        for l in self.model.layers:
            t.add_row([str(c), l.name, l.input_shape, l.output_shape, str(l.trainable)])
            c += 1
        print(t)


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
    activation_precision=1,  # BNN
    kernel_precision=1,  # BNN
    enable_history=False,
    clean_history=not False,
    optimizer=None,  # Adam will be used with the l_rate as a learning rate
    loss="categorical_crossentropy",
)
