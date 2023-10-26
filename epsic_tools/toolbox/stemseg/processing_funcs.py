# tensorflow
import pkg_resources
#pkg_resources.require("TensorFlow==2.3.0")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import hyperspy.api as hs

import time

# check version
print('Using TensorFlow v%s' % tf.__version__)
acc_str = 'accuracy' if tf.__version__[:2] == '2.' else 'acc'


import numpy as np
import matplotlib.pyplot as plt

# need certainty to explain some of the results
import random as python_random
python_random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

import numpy as np
import hyperspy.api as hs
import pyxem as pxm
import matplotlib.pyplot as plt
import os
import sys

from random import random
from ipywidgets import interact, interactive, fixed, interact_manual
from matplotlib.cm import get_cmap


from pyxem.libraries.calibration_library import CalibrationDataLibrary
from pyxem.generators.calibration_generator import CalibrationGenerator

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift

import math

import warnings
warnings.simplefilter(action='ignore')
plt.rcParams.update({'figure.max_open_warning': 0})
warnings.filterwarnings("ignore")

#!git clone https://github.com/ePSIC-DLS/epsic_tools
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, 'epsic_tools'))
#import epsic_tools.api as epsic

from sklearn import manifold, datasets
from functools import partial
from mpl_toolkits.mplot3d import Axes3D

from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.mixture import GaussianMixture as GM
from sklearn.feature_selection import mutual_info_classif as MIC

import sklearn.metrics.cluster as clust
from skimage.transform import resize

import pathlib
import hyperspy.api as hs
import pyxem as pxm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.manifold import MDS

from tensorflow import keras
from tensorflow.keras import layers


###### Functions ####

def null_manip(d):
    return d

def data_manip_lowq_resized(d, central_box = 256, bs = 256):
    pxc, pyc = d.shape[1]//2, d.shape[2]//2 
    pxl, pxu = pxc - central_box//2, pxc + central_box//2 
    pyl, pyu = pyc - central_box//2, pyc + central_box//2 
    
    d = d[:, pxl:pxu, pyl:pyu]
    if type(d) != np.ndarray:
        print('dask to numpy')
        d = d.compute()
        print('dask to numpy done')
    print('started data manipulations')
    #d = resize(d,(d.shape[0],128,128))
    print('resized')
    d = d.astype('float32')
    for i in range(d.shape[0]):
        d_max = np.max(d[i])
        if d_max <= 0:
            d_max = 1
        d[i] = d[i]/d_max
    d = batch_resize(d, bs)
    scaler = np.log(1001)
    return np.log((d*1000)+1)/scaler 


### PCA #####

def find_PCA_dims(data, nav_skips = 4, nav_crop = 2):
    data = data.inav[nav_crop:-nav_crop,nav_crop:-nav_crop].inav[::nav_skips,::nav_skips] #scale down dataset to a smaller size
    data.compute()
    data = pxm.signals.ElectronDiffraction2D(data.data.astype('float32'))
    print(data)
    data.decomposition(True, algorithm = 'SVD')
    data.plot_explained_variance_ratio()
    data = None #make sure the memory is freed up
    
    
def get_PCA_signal(comp_data,inspect_data, ndims, nav_skips = 4, nav_crop = 2, decomp_thresh = 1e-1, norm= True, inv = False, use_mask= True, inc_nav = True):
    cent = comp_data.inav[nav_crop:-nav_crop,nav_crop:-nav_crop].inav[::nav_skips,::nav_skips]
    try:
        cent.compute()
    except:
        pass
    cent = pxm.signals.ElectronDiffraction2D(cent.data.astype('float32'))
    print(cent)
    cent.decomposition(True, algorithm='NMF', output_dimension=ndims)
    if use_mask == True:
        if inv == False:
            masks = cent.get_decomposition_factors() > decomp_thresh
        else:
            masks = cent.get_decomposition_factors() < decomp_thresh
        mask_data = masks.data
        #Does the NMF

        #Apply each N masks in turn to every pattern and sum the result. Will have a signal with N dimensions
        # with each dimension giving the intensity of one of the masked regions
        sig = []
        for i in range(mask_data.shape[0]):
            def apply_mask(x):
                return x * mask_data[i]
            prodata = inspect_data.deepcopy()
            process = prodata.map(apply_mask).T.sum()
            process.compute()
            sig.append(process.data)
    elif use_mask == False:
        print('no mask')
        decomp_data = cent.get_decomposition_factors().data
        print('got data')
        decomp_data = decomp_data/decomp_data.max(axis = (1,2))[:,None, None]
        print('normalised')
        if inv == True:
            decomp_data = np.abs(decomp_data- 1)
            print('inverted')
        sig = [(inspect_data.deepcopy()*decomp_data[i]).sum(axis=(2,3)).data[None,:,:] for i in range(decomp_data.shape[0])]
        #for i in range(decomp_data.shape[0]):
        #    print(i)
        #    prodata = inspect_data.deepcopy()
        #    process = (prodata*decomp_data[i]).sum(axis=(2,3))
        #    #process.compute()
        #    sig.append(process.data[None,:,:].astype('float32'))
        #    print(sig[0].shape)
            
            
    s = np.concatenate(sig, axis = 0)
    print('concat')
    s = np.moveaxis(s, 0,-1)
    if norm == True:
        print('norm start')
        ns = (s - s.min(axis = (0,1))[None, None, :])/s.max(axis = (0,1))[None, None, :]
        ns = ns/np.linalg.norm(ns, axis = -1)[:, :, None]
        print('norm finished')
    else:
        ns = s
    ns = np.where(np.isnan(ns),0,ns)
    if inc_nav == True:
        s0, s1 = inspect_data.data.shape[0], inspect_data.data.shape[1]
        nav_space0 = np.repeat(np.arange(0, s0)[:,None], s1, axis = 1)
        nav_space1 = np.repeat(np.arange(0, s1)[:,None], s0, axis = 1).T

        nav_space = np.concatenate([nav_space0[:,:,None], nav_space1[:,:, None]], axis = 2)

        nav_space.shape

        fns = flatten_nav(nav_space).astype('float32')

        ins = flatten_nav(inspect_data.data).mean(axis=(1,2))[:,None]

        ins = ins-ins.min()

        ins = (ins)/(ins.max())
        fns[:,0] = fns[:,0]/(2*fns[:,0].max())
        fns[:,1] = fns[:,1]/(2*fns[:,1].max())

        #r_space = np.concatenate([fns, ins], axis = 1)
        r_space = ins.reshape((s0,s1, 1))
        ns = np.concatenate([ns, r_space], axis = -1)
    return ns,cent

def seperate_background(ns):
    flat_s = flatten_nav(ns)
    hdc = GM(2).fit_predict(flat_s)
    return flat_s, hdc


def PCA_reduced_representation(ns, perplexity, niters, lr, ee, pre_seperate):
    #can try to pre-seperate out the sample from the background, this can help with speeding up TSNE
    if pre_seperate == True:
        flat_s, sep_clusts  = seperate_background(ns)
        #assuming there will be more background than sample you assume the smaller set is the sample
        if len(np.where(sep_clusts==0)[0]) < len(np.where(sep_clusts==1)[0]):
            sample_cat = 0
        else:
            sample_cat = 1
        flat_cs = flat_s[np.where(sep_clusts==sample_cat)[0]]
    else:
        sep_clusts = None
        flat_cs = flatten_nav(ns)
    print(flat_cs.shape)
    rep = TSNE(n_components=2, perplexity=perplexity, verbose=2, n_iter=niters, learning_rate = lr, early_exaggeration= ee).fit_transform(flat_cs)
    return rep, sep_clusts


def PCA_clustering(raw, rep, sep_clusts, clust_func, func_params):
    flat_raw = flatten_nav(raw.data)
    if clust_func == 'gm':
        from sklearn.mixture import GaussianMixture as GM
        c = GM(func_params[0]).fit_predict(rep)
    if clust_func == 'dbscan':
        from sklearn.cluster import DBSCAN as DB
        c = DB(eps = func_params[0], min_samples = func_params[1]).fit_predict(rep)
    print(c.max())
    #Deals with reindexing to accomodate the pre-removed background
    if np.all(sep_clusts != None):
        if len(np.where(sep_clusts==0)[0]) < len(np.where(sep_clusts==1)[0]):
            sample_cat = 0
        else:
            sample_cat = 1
        sep_clusts[np.where(sep_clusts==sample_cat)[0]]=c+sep_clusts.max()+1
        seg= reindex_clusters(sep_clusts)
    #If background wasn't removed - do nothing
    else:
        seg = c
    fig, patts = show_patterns(flat_raw, seg)
    return seg, fig, patts
        
#### VAE #####
        
        
def get_best_model(meta_data, resize_latent,beta, mean_thresh=0.05):
    model_type = meta_data.model.type
    model_path = meta_data.model.path
    latent_dim = meta_data.model.dims
    
    n_img = 128
    
    dataset_root = str(model_path)
    if Path(dataset_root).exists()!= True:
        Path(dataset_root).mkdir()
    chkpoint_fileroot = f'{dataset_root}/{latent_dim}/'
    if Path(chkpoint_fileroot).exists()!= True:
        Path(chkpoint_fileroot).mkdir()
    chkpoint_filepath = chkpoint_fileroot + 'chk-{epoch:02d}-{val_loss:.5f}.ckpt'
    
    best_epoch = 0
    best_chk_ind = None
    for ind, chk_path in enumerate((model_path/f'{latent_dim}').ls()):
        try:
            epoch = int((str(chk_path).split('.ckpt')[0]).split('-')[-2])
        except:
            continue
        if epoch > best_epoch:
            best_epoch = epoch
            best_chk_ind = ind
            
    if best_epoch == 0:
        print('No model found - please pretrain one')
        return None
    
    else:
        
        best_chk = (model_path/f'{latent_dim}').ls()[best_chk_ind]
        best_chk = f"{str(best_chk).split('.ckpt')[0]}.ckpt"

        # sampling z with (z_mean, z_log_var)
        class Sampling(layers.Layer):
            def call(self, inputs):
                z_mean, z_log_var = inputs
                epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon

        if model_type == 'nn':
            # build the encoder
            image_input = keras.Input(shape=(n_img, n_img))
            x = layers.Flatten()(image_input)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.Dense(16, activation="relu")(x)
            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z_output = Sampling()([z_mean, z_log_var])
            encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output])

            # build the decoder
            z_input = keras.Input(shape=(latent_dim,))
            x = layers.Dense(16, activation="relu")(z_input)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dense(n_img * n_img, activation="sigmoid")(x)
            image_output = layers.Reshape((n_img, n_img))(x)
            decoder_VAE = keras.Model(z_input, image_output)

        if model_type == 'cnn':
            image_input = keras.Input(shape=(n_img, n_img,1), name = 'enc_input')
            x = layers.Conv2D(16,3, strides = 2, activation='relu',padding='same', input_shape=image_input.shape, name = 'enc_conv1')(image_input)
            x = layers.Conv2D(32,3, strides = 2, activation='relu',padding='same', name = 'enc_conv2')(x)
            x = layers.Flatten()(x)
            x = layers.Dense(32, activation='relu', name = 'enc_d1')(x)
            x = layers.Dense(16, activation="relu", name = 'enc_d2')(x)
            z_mean = layers.Dense(latent_dim, name="z_mean")(x)
            z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
            z_output = Sampling()([z_mean, z_log_var])
            encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output])

            z_input = keras.Input(shape=(latent_dim,), name = 'dec_input')
            x = layers.Dense(16, activation="relu", name = 'dec_d1')(z_input)
            x = layers.Dense(32, activation="relu", name = 'dec_d2')(x)
            x = layers.Dense(32 * 32 *32, activation="relu", name = 'dec_d3')(x)
            x = layers.Reshape((32, 32,32))(x)
            x = layers.Conv2DTranspose(16,3, strides = 2, activation='relu',padding='same', name = 'dec_conv1')(x)
            image_output = layers.Conv2DTranspose(1,3, strides = 2, activation='sigmoid',padding='same', name = 'dec_conv2')(x)
            #image_output = layers.Conv2DTranspose(16,3, strides = 2, activation='sigmoid',padding='same')
            #image_output = layers.Reshape((n_img, n_img,1))(x)
            decoder_VAE = keras.Model(z_input, image_output)

        # VAE class
        class VAE(keras.Model):
            # constructor
            def __init__(self, encoder, decoder,beta, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder
                self.beta = beta
                

            # customise train_step() to implement the loss 
            def train_step(self, x):
                if isinstance(x, tuple):
                    x = x[0]
                with tf.GradientTape() as tape:
                    # encoding
                    z_mean, z_log_var, z = self.encoder(x)
                    # decoding
                    x_prime = self.decoder(z)
                    # reconstruction error by binary crossentropy loss
                    reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime)) * n_img * n_img
                    # KL divergence
                    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    # loss = reconstruction error + KL divergence
                    loss = reconstruction_loss + self.beta*kl_loss
                # apply gradient
                grads = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                # return loss for metrics log
                return {"loss": loss,
                        "reconstruction_loss": reconstruction_loss,
                        "kl_loss": kl_loss}


            def call(self, x):
                if isinstance(x, tuple):
                    x = x[0]
                # encoding
                z_mean, z_log_var, z = self.encoder(x)
                # decoding
                x_prime = self.decoder(z)
                return x_prime
        # build the VAE
        vae_model = VAE(encoder_VAE, decoder_VAE, beta)

        # compile the VAE
        vae_model.compile(optimizer=keras.optimizers.Adam(), loss = 'binary_crossentropy')

        print(best_chk)
        vae_model.load_weights(best_chk)
        
        
        if resize_latent != None:
            latent_dim = resize_latent
            if model_type == 'nn':
                # build the encoder
                image_input = keras.Input(shape=(n_img, n_img))
                x = layers.Flatten()(image_input)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.Dense(16, activation="relu")(x)
                z_mean = layers.Dense(latent_dim, name="z_mean")(x)
                z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
                z_output = Sampling()([z_mean, z_log_var])
                encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output])

                # build the decoder
                z_input = keras.Input(shape=(latent_dim,))
                x = layers.Dense(16, activation="relu")(z_input)
                x = layers.Dense(128, activation="relu")(x)
                x = layers.Dense(n_img * n_img, activation="sigmoid")(x)
                image_output = layers.Reshape((n_img, n_img))(x)
                decoder_VAE = keras.Model(z_input, image_output)

            if model_type == 'cnn':
                image_input = keras.Input(shape=(n_img, n_img,1), name = 'enc_input')
                x = layers.Conv2D(16,3, strides = 2, activation='relu',padding='same', input_shape=image_input.shape, name = 'enc_conv1')(image_input)
                x = layers.Conv2D(32,3, strides = 2, activation='relu',padding='same', name = 'enc_conv2')(x)
                x = layers.Flatten()(x)
                x = layers.Dense(32, activation='relu', name = 'enc_d1')(x)
                x = layers.Dense(16, activation="relu", name = 'enc_d2')(x)
                z_mean = layers.Dense(latent_dim, name="z_mean")(x)
                z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
                z_output = Sampling()([z_mean, z_log_var])
                encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output])

                z_input = keras.Input(shape=(latent_dim,), name = 'dec_input')
                x = layers.Dense(16, activation="relu", name = 'dec_d1')(z_input)
                x = layers.Dense(32, activation="relu", name = 'dec_d2')(x)
                x = layers.Dense(32 * 32 *32, activation="relu", name = 'dec_d3')(x)
                x = layers.Reshape((32, 32,32))(x)
                x = layers.Conv2DTranspose(16,3, strides = 2, activation='relu',padding='same', name = 'dec_conv1')(x)
                image_output = layers.Conv2DTranspose(1,3, strides = 2, activation='sigmoid',padding='same', name = 'dec_conv2')(x)
                #image_output = layers.Conv2DTranspose(16,3, strides = 2, activation='sigmoid',padding='same')
                #image_output = layers.Reshape((n_img, n_img,1))(x)
                decoder_VAE = keras.Model(z_input, image_output)
                
            new_vae_model = VAE(encoder_VAE, decoder_VAE,beta)
            new_vae_model.compile(optimizer=keras.optimizers.Adam(), loss = 'binary_crossentropy')
            
            for l in range(6):
                w = vae_model.get_layer(index = 0).get_layer(index=l).get_weights()
                nl = new_vae_model.get_layer(index = 0).get_layer(index=l).set_weights(w)
                
            for l in range(2,7):
                w = vae_model.get_layer(index = 1).get_layer(index=l).get_weights()
                nl = new_vae_model.get_layer(index = 1).get_layer(index=l).set_weights(w)
                
            out_model = new_vae_model
        else:
            out_model = vae_model

        return out_model


def process_data(dp, nav_skip = 6):
    dt = hs.load(dp,lazy=True).inav[::nav_skip,::nav_skip]
    dt = dt.data
    dt = flatten_nav(dt)
    dt = ml_resize(dt)
    cb = np.where(np.mean(np.moveaxis(dt,0,-1),-1)>1,0,1)
    dt = dt*cb
    dt /= dt.max()
    return dt



########



def pretrain_model(dps, meta_data, mean_thresh=0.05, max_epochs = 100, nav_skip = 6):
    model_type = meta_data.model.type
    model_path = meta_data.model.path
    latent_dim = meta_data.model.dims
    bs = meta_data.model.bs
    n_img = 128
    
    dataset_root = str(model_path)
    if Path(dataset_root).exists()!= True:
        Path(dataset_root).mkdir()
    chkpoint_fileroot = f'{dataset_root}/{latent_dim}/'
    if Path(chkpoint_fileroot).exists()!= True:
        Path(chkpoint_fileroot).mkdir()
    chkpoint_filepath = chkpoint_fileroot + 'chk-{epoch:02d}-{val_loss:.5f}.ckpt'

    # sampling z with (z_mean, z_log_var)
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    if model_type == 'nn':
        # build the encoder
        image_input = keras.Input(shape=(n_img, n_img))
        x = layers.Flatten()(image_input)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z_output = Sampling()([z_mean, z_log_var])
        encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output])

        # build the decoder
        z_input = keras.Input(shape=(latent_dim,))
        x = layers.Dense(16, activation="relu")(z_input)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(n_img * n_img, activation="sigmoid")(x)
        image_output = layers.Reshape((n_img, n_img))(x)
        decoder_VAE = keras.Model(z_input, image_output)
        
    if model_type == 'cnn':
        image_input = keras.Input(shape=(n_img, n_img,1), name = 'enc_input')
        x = layers.Conv2D(16,3, strides = 2, activation='relu',padding='same', input_shape=image_input.shape, name = 'enc_conv1')(image_input)
        x = layers.Conv2D(32,3, strides = 2, activation='relu',padding='same', name = 'enc_conv2')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation='relu', name = 'enc_d1')(x)
        x = layers.Dense(16, activation="relu", name = 'enc_d2_t')(x)
        z_mean = layers.Dense(latent_dim, name="z_mean_t")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var_t")(x)
        z_output = Sampling()([z_mean, z_log_var])
        encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output])

        z_input = keras.Input(shape=(latent_dim,), name = 'dec_input_t')
        x = layers.Dense(16, activation="relu", name = 'dec_d1_t')(z_input)
        x = layers.Dense(32, activation="relu", name = 'dec_d2')(x)
        x = layers.Dense(32 * 32 *32, activation="relu", name = 'dec_d3')(x)
        x = layers.Reshape((32, 32,32))(x)
        x = layers.Conv2DTranspose(16,3, strides = 2, activation='relu',padding='same', name = 'dec_conv1')(x)
        image_output = layers.Conv2DTranspose(1,3, strides = 2, activation='sigmoid',padding='same', name = 'dec_conv2')(x)
        #image_output = layers.Conv2DTranspose(16,3, strides = 2, activation='sigmoid',padding='same')
        #image_output = layers.Reshape((n_img, n_img,1))(x)
        decoder_VAE = keras.Model(z_input, image_output)

    # VAE class
    class VAE(keras.Model):
        # constructor
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder

        # customise train_step() to implement the loss 
        def train_step(self, x):
            if isinstance(x, tuple):
                x = x[0]
            with tf.GradientTape() as tape:
                # encoding
                z_mean, z_log_var, z = self.encoder(x)
                # decoding
                x_prime = self.decoder(z)
                # reconstruction error by binary crossentropy loss
                reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime)) * n_img * n_img
                # KL divergence
                kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                # loss = reconstruction error + KL divergence
                loss = reconstruction_loss + kl_loss
            # apply gradient
            grads = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            # return loss for metrics log
            return {"loss": loss,
                    "reconstruction_loss": reconstruction_loss,
                    "kl_loss": kl_loss}


        def call(self, x):
            if isinstance(x, tuple):
                x = x[0]
            # encoding
            z_mean, z_log_var, z = self.encoder(x)
            # decoding
            x_prime = self.decoder(z)
            return x_prime
    # build the VAE
    vae_model = VAE(encoder_VAE, decoder_VAE)

    # compile the VAE
    vae_model.compile(optimizer=keras.optimizers.Adam(), loss = 'binary_crossentropy')
    m = process_data(dps[0], nav_skip)
    for dp in dps[1:]:
        print(dp)
        dt = process_data(dp, nav_skip)
        print(dt.shape)
        m = np.concatenate((m,dt))
    print('array')
    adt = flatten_nav(m)
    #mean_vals = m.mean(axis =-1).mean(axis=-1)
    print('all loaded', m.shape)

    #n = m[np.where(mean_vals > mean_thresh)[0]]
    #m = n

    idxs = np.random.permutation(range(m.shape[0]))
    cut = int(np.round(0.8 * m.shape[0],0))

    train_data = m[idxs[:cut]]
    valid_data = m[idxs[cut:]]

    if model_type == 'cnn':
        train_data = train_data[:,:,:,np.newaxis]
        valid_data = valid_data[:,:,:,np.newaxis]

        print('axis added', train_data.shape)

    # load dataset

    # normalise images
    #train_data = train_data / m.max()
    #valid_data = valid_data / m.max()
    
    #print('normalised')
    
    
    tds = tf.data.Dataset.from_tensor_slices((train_data, train_data))
    vds = tf.data.Dataset.from_tensor_slices((valid_data, valid_data))
    
    tds = tds.batch(bs)
    vds = vds.batch(bs)

    print('batched')


    chkpoint_model = tf.keras.callbacks.ModelCheckpoint(
        filepath = chkpoint_filepath,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        save_freq="epoch",
        options=None)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.000001,
        patience=50,
        verbose=1,
        mode="min")


    vae_model.fit(tds, validation_data=vds, epochs=max_epochs, callbacks = [chkpoint_model,early_stop])
    
def preprocess_learning_data(dps, save_path):
    m = process_data(dps[0], nav_skip)
    for dp in dps[1:]:
        print(dp)
        dt = process_data(dp, nav_skip)
        print(dt.shape)
        m = np.concatenate((m,dt))
    m.to_hdf5(save_path)
    print('array')


def prepare_model(meta_data, resize_latent, beta):
    model_type = meta_data.model.type
    model_path = meta_data.model.path
    latent_dim = meta_data.model.dims
    
        
    

    # build the VAE
    vae_model = get_best_model(meta_data, resize_latent, beta)
    
    vae_model.encoder.get_layer(index = 1).trainable = False
    vae_model.encoder.get_layer(index = 2).trainable = False
    #vae_model.encoder.get_layer(index = 3).trainable = False
    #vae_model.encoder.get_layer(index = 4).trainable = False
    vae_model.decoder.get_layer(index = -1).trainable = False
    vae_model.decoder.get_layer(index = -2).trainable = False
    #vae_model.decoder.get_layer(index = -3).trainable = False
    #vae_model.decoder.get_layer(index = -4).trainable = False
        
        
    
    

    # compile the VAE
    #vae_model.compile(optimizer=keras.optimizers.Adam(), loss = 'binary_crossentropy')
    
    return vae_model


def get_learning_data(raw_data,meta_data, dmanip, loaded_lazily, input_data):
    model_type = meta_data.model.type
    
    dt = raw_data.data
    dt = flatten_nav(dt)
    if np.all(input_data !=None):
        m = input_data
    else:
        if dmanip == None:
            adt = ml_resize(dt)
            cb = np.where(np.mean(np.moveaxis(adt,0,-1),-1)>1,0,1)
            m = adt*cb
            if loaded_lazily == True:
                m = m.compute()
        else:
            print('starting data manipulations')
            m = dmanip(dt)
            if loaded_lazily == True:
                print('computing data manipulations')
                m = m.compute()
                print('computed')
    #mean_vals = m.mean(axis =-1).mean(axis=-1)

    #n = m[np.where(mean_vals > mean_thresh)[0]]


    #m = n

    idxs = np.random.permutation(range(m.shape[0]))
    cut = int(np.round(0.8 * m.shape[0],0))

    train_data = m[idxs[:cut]]
    valid_data = m[idxs[cut:]]

    if model_type == 'cnn':
        train_data = train_data[:,:,:,np.newaxis]
        valid_data = valid_data[:,:,:,np.newaxis]
    

    print('data loaded for training', train_data.shape)

    # load dataset
    (train_images, train_labels), (test_images, test_labels) = [train_data, train_data], [valid_data, valid_data]

    # normalise images
    #train_images = train_images / m.max()
    #test_images = test_images / m.max()

    
    return train_images, test_images

def transfer_learn(model, meta_data, dp, epoch, raw_data, use_full_ds, dmanip, lazy_load = True,input_data =None):
    model_type = meta_data.model.type
    model_path = meta_data.model.path
    latent_dim = meta_data.model.dims
    bs = meta_data.model.bs
    
    if use_full_ds == True:
        all_data = hs.load(dp, lazy = lazy_load)
    else:
        all_data = raw_data
    
    train_images, test_images = get_learning_data(all_data,meta_data, dmanip, not lazy_load, input_data)
    tds = tf.data.Dataset.from_tensor_slices((train_images, train_images))
    vds = tf.data.Dataset.from_tensor_slices((test_images, test_images))
    
    tds = tds.batch(bs)
    vds = vds.batch(bs)
    
    def custom_loss(x,y,n_img=128):
        return tf.reduce_mean(keras.losses.binary_crossentropy(x, y)) * n_img * n_img
    
    model.compile(optimizer=keras.optimizers.Adam(), loss = custom_loss)
    
    

    

    print('normalised')



    model.fit(tds, validation_data=vds, epochs=epoch, verbose = 2)
    return model

##### MI ###
    
    


def find_all_peaks(raw_data):
    maxsig= np.max(flatten_nav(raw_data.data),0)
    scaled_max = maxsig/maxsig.max()

    scaled_max *= 64
    scaled_max = scaled_max.astype('uint8')

    sm = scaled_max.compute()

    max_signal = pxm.signals.ElectronDiffraction2D(sm)
    peaks = max_signal.find_peaks(method = 'minmax',interactive=True, distance=5.30, threshold = 8)
    return peaks
    
def peakwise_signal(peaks, raw_data, view_df = None, rsize = 2, scale_max = None):
    odata = pxm.signals.ElectronDiffraction2D(raw_data.data)
    print(type(odata))
    px, py = peaks.data.T
    npeaks=peaks.data.shape[0]

    dfmaps = np.array(())
    for peak in peaks.data:
        print(peak)
        ppx, ppy = peak
        roi = hs.roi.CircleROI(cx=ppy, cy=ppx, r=10, r_inner=0)
        dfmaps = np.append(dfmaps,(odata.get_integrated_intensity(roi)))

    n_dfmaps = np.ma.filled(dfmaps,0)
    
    nshape = raw_data.data.shape
    
    #for i in range(n_dfmaps.shape[0]):
    #    axis_mean = np.mean(n_dfmaps[i])
    #    print(n_dfmaps[i],n_dfmaps[i].shape, axis_mean,i)
    #    n_dfmaps[i] -= axis_mean
    

    clustmap = n_dfmaps.reshape(npeaks,nshape[0],nshape[1]).astype('float32')

    for i in range(clustmap.shape[0]):
        axis_sum = np.sum(clustmap[i])
        clustmap[i] *= 1/axis_sum

    movemap = np.moveaxis(clustmap,0,-1)
            
    dim_clust = pxm.signals.ElectronDiffraction1D(movemap)
    return dim_clust     
    
    
def correlated_peaks_tsne(raw_data, peaks, metadata, mutual_threshold,without_mi,perplexity, nruns, lr, ee, verbose=True, show_figs=False, eg_ind=10):
    
    dim_clust = peakwise_signal(peaks, raw_data)

    r_clm = dim_clust.data.reshape(int(dim_clust.data.size/dim_clust.data.shape[-1]),dim_clust.data.shape[-1])

    vpdf = r_clm.copy()
    
    print(vpdf.shape)

    npeaks = vpdf.shape[1]
    if verbose == True:
        print(f'Number of peaks found: {npeaks}')

    fig1 = plt.figure()
    plt.plot(range(npeaks),vpdf[eg_ind])
    plt.title(f'Peakwise Signal for pixel {eg_ind}')
    if show_figs == False:
        plt.close()

    bws = np.where(vpdf >0.0001,1,0)

    square_shape = (int(npeaks/asap(npeaks)),asap(npeaks))
    
    print(square_shape)

    imgrepr = vpdf[eg_ind].reshape(square_shape)
    bimgrepr = bws[eg_ind].reshape(square_shape)
    fig2, axs = plt.subplots(2,1)
    axs[0].imshow(imgrepr)
    axs[0].axis('off')
    axs[1].imshow(bimgrepr)
    axs[1].axis('off')
    if show_figs == False:
        plt.close()

    fig3,axs = plt.subplots(square_shape[0], square_shape[1])
    mic_d = []
    for ind in range(npeaks):
        mi = MIC(bws, np.moveaxis(bws,0,-1)[ind], discrete_features=True)
        mic_d.append(mi)
        if square_shape[-1] != 1:
            axs[np.unravel_index(ind,square_shape)].plot(np.arange(0,npeaks),mi)
            axs[np.unravel_index(ind,square_shape)].set_xticks([])
            axs[np.unravel_index(ind,square_shape)].set_yticks([])
        else:
            axs[ind].plot(np.arange(0,npeaks),mi)
            axs[ind].set_xticks([])
            axs[ind].set_yticks([])
    if show_figs == False:
        plt.close()

    mic_ar = np.asarray(mic_d)

    fig4 = plt.figure()
    plt.imshow(mic_ar)
    plt.title('Mutual Information in the peakwise signals')
    if show_figs == False:
        plt.close()

    overlap = np.moveaxis(np.array(np.where(mic_ar>mutual_threshold)),0,1)
    
    print(overlap)

    swb = np.moveaxis(vpdf.copy(),0,1)

    dependent_signals = []
    redundancy_dict = {}
    for coo in overlap:
        cx, cy = coo
        if cx != cy:
            try:
                redundancy_dict[cy].append(cx)
            except:
                redundancy_dict[cy] = [cx]
            try:
                redundancy_dict[cx].index(cy)
            except:
                dependent_signals.append(coo)
    dependent_signals = np.asarray(dependent_signals)
    print(dependent_signals)

    current_pos = None
    current_sig = None
    remaining_signals = list(range(npeaks))
    paired_signals = []
    for each in dependent_signals:
        try:
            remaining_signals.pop(remaining_signals.index(each[0]))
        except:
            continue
        try:
            remaining_signals.pop(remaining_signals.index(each[1]))
        except:
            continue
        if each[0] == current_pos:
            current_sig += vpdf[:,each[1]]
        else:
            paired_signals.append(current_sig)
            current_sig = vpdf[:,each[0]]+vpdf[:,each[1]]
            current_pos = each[0]
    #need to decide whether to include this
    #for remaining in remaining_signals:
    #    paired_signals.append(vpdf[:,remaining])
    if paired_signals != []:
        paired_signals.pop(0)
        paired_signals = np.moveaxis(np.asarray(paired_signals),0,1)

        red_npeaks = paired_signals.shape[1]

        bwps = np.where(paired_signals>0.0005,1,0)

        fig5, axs = plt.subplots(3,1)
        n =eg_ind
        axs[0].plot(np.arange(npeaks),vpdf[n])
        axs[1].plot(np.arange(npeaks),bws[n])
        axs[2].plot(np.arange(red_npeaks),paired_signals[n])
        if show_figs == False:
            plt.close()
            
    else:
        print('no paired signals at current threshold')
        fig5, axs = plt.subplots(3,1)
        paired_signals = vpdf
        
    if without_mi == False:

        mds = manifold.TSNE(2,init='pca',verbose = 1, perplexity=perplexity, n_iter = nruns, learning_rate = lr, early_exaggeration= ee,n_jobs = -1).fit_transform(paired_signals)
        outsig = paired_signals
    else:
        mds = manifold.TSNE(2,init='pca',verbose = 1,perplexity=perplexity, n_iter = nruns, learning_rate = lr, early_exaggeration= ee,n_jobs = -1).fit_transform(vpdf)
        outsig = vpdf
    tX,tY = np.moveaxis(mds,0,1)
    fig6 = plt.figure()
    plt.scatter(tX,tY, s=1, alpha = 0.1)
    plt.title('t-SNE of MI-reduced peak signal')
    if show_figs == False:
        plt.close()

    bics = []
    for ncom in range(1,20):
        print(ncom)
        gmm = GM(n_components=ncom).fit(mds)
        bics.append(gmm.bic(mds))

    fig7 = plt.figure()
    plt.plot(range(1,20), bics)
    plt.title('Bayesian Information Criteria for different cluster numbers')
    if show_figs == False:
        plt.close()
        
    return mds, outsig, [fig1, fig2, fig3, fig4, fig5, fig6,fig7]

def get_map_from_rep(rep, metadata, adt, recover_locs, clust_func, func_params, show_figs = False):
    
    tX,tY = np.moveaxis(rep,0,1)
    if clust_func == 'gm':
        from sklearn.mixture import GaussianMixture as GM
        scaling_factor = rep[:,0].max() - rep[:,0].min() 
        gm_model = GM(func_params[0]).fit(rep/scaling_factor)
        c = gm_model.predict(rep/scaling_factor)
        cprob = gm_model.predict_proba(rep/scaling_factor)
    if clust_func == 'dbscan':
        from sklearn.cluster import DBSCAN as DB
        gm_model = DB(eps = func_params[0], min_samples = func_params[1]).fit(rep)
        c = gm_model.fit_predict(rep)
        cprob = np.ones_like(c)
    fig1 = plt.figure()
    plt.scatter(tX,tY,c=c, s=1, cmap='turbo')
    plt.colorbar()
    if show_figs == False:
        plt.close()
        
    navplane = adt.data[:,:,0,0]

    blank = np.zeros(navplane.shape)
    for ind, coords in enumerate(recover_locs):
        xcoord, ycoord = coords
        blank[xcoord, ycoord] += c[ind]

    fig2 = plt.figure()
    plt.imshow(blank,'turbo')
    plt.colorbar()
    if show_figs == False:
        plt.close()
    comps = c.max()
    fig5,axs = plt.subplots(comps)
    flat_raw = flatten_nav(adt.data)
    for ind in range(comps):
        print(ind)
        clocs = np.where(c==ind)[0]    
        try:
            diff_patt = flat_raw[clocs]
            diff_patt = np.max(np.asarray(diff_patt),0)
        except:
            diff_patt = np.zeros_like(flat_raw[0])

        axs[ind].imshow(diff_patt)
        axs[ind].axis('off')
        axs[ind].set_title(ind+1)
    if show_figs == False:
        plt.close()
        

        
    return [fig1, fig2, fig5], blank, c, cprob


##### compare maps ####


def f(x):
    return x   

def get_refined_plots(sample,nclust, tape):

    info = {}
    for x in range(nclust+1):
        info[x] = []
    for key,val in tape.items():
        try:
            info[val[-1]].append(key)
        except:
            continue


    blank_map = np.zeros_like(sample.mi_map_data[0])
    blank_regions = []
    region_patterns = []
    mean_patterns = []
    for key in info.keys():
        region = []
        pattern = []
        for each in info[key]:
            region.append(sample.map_overlaps_data[each])
            pattern.append(sample.region_patterns[each])
        region = np.asarray(region)
        pattern = np.asarray(pattern)
        blank_map += np.where(region.sum(axis=0)>=1,key,0)
        try:
            blank_regions.append(np.where(region.sum(axis=0)>=1,1,0))
        except:
            blank_regions.append([])
        try:
            region_patterns.append(pattern.max(0))
        except:
            region_patterns.append([])
        try:
            mean_patterns.append(pattern.mean(0))
        except:
            mean_patterns.append([])

    refined_map = plt.figure()
    plt.imshow(blank_map, cmap = 'turbo')
    plt.axis('off')
    plt.colorbar()
    
    patt_shape = [int(6/asap(nclust+1)),asap(nclust+1)]

    fig2,ax = plt.subplots(patt_shape[0], patt_shape[1])
    for ind, patts in enumerate(region_patterns):
        if patts == []:
            patts = np.zeros((255,254))
        ax[np.unravel_index(ind, patt_shape)].imshow(patts[50:-50,50:-50],cmap='turbo')
        ax[np.unravel_index(ind, patt_shape)].set_xticks([])
        ax[np.unravel_index(ind, patt_shape)].set_yticks([])
        ax[np.unravel_index(ind, patt_shape)].set_title(f'{ind}')
        
    return refined_map, blank_map, fig2, region_patterns, mean_patterns




### misc ####

def expand_clustering(flat_s, hdc, fixed_labels, nclusts):
    old_nclusts = list(range(hdc.max()+1))
    if type(fixed_labels) != list:
        fixed_labels = [fixed_labels]
    n_fixed_labels = len(fixed_labels)
    for each in fixed_labels:
        old_nclusts.remove(each)
    tbsorted = np.array(())
    for each in old_nclusts:
        tbsorted= np.append(tbsorted,np.where(hdc==each)[0]).astype('int')
    flat_s_2 = flat_s[tbsorted]
    hdc2 = GM(nclusts).fit_predict(flat_s_2)
    out = hdc.copy()
    fixed_labels.sort()
    for ind, each in enumerate(fixed_labels):
        out = np.where(out ==each, ind, out)
    for each in range(nclusts):
        out[tbsorted[np.where(hdc2==each)[0]]]=each + n_fixed_labels
    return out

def contract_clustering(hdc, fixed_labels):
    scalar = hdc.max() + 1
    new = hdc.copy() + scalar
    current = list(range(scalar))
    for each in fixed_labels:
        new = np.where(new == each+scalar, 0, new)
        current.remove(each)
    for ind, other in enumerate(current):
        new = np.where(new == other+scalar, ind+1, new)
    return new


from numba import njit
@njit
def quick_patterns(empty,indx, indy, pattx, patty,i, stack_arr):
    empty[indx*pattx:(indx*pattx+pattx), indy*patty:(indy*patty+patty)] = stack_arr[i]
    return empty

def tile_stack(stack_arr):
    s = stack_arr.shape[0]
    for i in range(1,int(np.round(s**0.5,0))+1):
        if s % i == 0:
            rsize = i
    shape = [int(s//rsize), int(rsize)]
    patt_size = stack_arr.shape[1:]
    empty = np.zeros((shape[0]*patt_size[0], shape[1]*patt_size[1]))
    print(empty.shape)
    for i in range(stack_arr.shape[0]):
        ind = np.unravel_index(i, shape)
        indx = int(ind[0])
        indy = int(ind[1])
        pattx = int(patt_size[0])
        patty = int(patt_size[1])
        quick_patterns(empty,indx, indy, pattx, patty,i, stack_arr)
    fig = plt.figure()
    plt.imshow(empty)
    plt.xticks([])
    plt.yticks([])
    return fig

def show_patterns(flat_raw,clust_map,method='max'):
    clust_map = clust_map.astype('int')
    flat_map = flatten_nav(clust_map)
    flat_raw = np.asarray(flat_raw)
    for n, i in enumerate(np.unique(clust_map)):
        patt = flat_raw[np.where(flat_map == i)[0]]
        try:
            if method =='max':
                patt_max = patt.max(0)
            if method =='mean':
                patt_max = patt.mean(0)
        except:
            patt_max = np.zeros(flat_raw.shape[1:])
        if n == 0:
            patts = patt_max[None,:,:]
        else:
            patts = np.concatenate((patts, patt_max[None,:,:]),axis = 0)    
    fig = tile_stack(patts)
    return fig, patts  

def reindex_clusters(c):
    current = []
    scale = c.max()+1
    out = c.copy()+scale
    for each in range(scale):
        if len(np.where(c==each)[0]) >0:
            current.append(each)
    for ind, each in enumerate(current):
        out = np.where(out == each +scale, ind, out)
    return out

    

def get_day(phase):
    return str(Path(f'/dls/e02/data/2021/mg28034-1/processing/Merlin/Calibrated/{phase}/').ls()[0]).split('/')[-1].split(' ')[0]

def flatten_nav(sig):
    shape = [sig.shape[0]*sig.shape[1]]
    for i in sig.shape[2:]:
        shape.append(i)
    return sig.reshape(shape)


def get_full_data_path(ts,data_path):
    for x, f in enumerate(data_path.ls()):
        try:
            index = str(f).index(ts)
            data_index = x
        except:
            pass
    dset_num = data_index
    data_options = [i for i in data_path.ls()[dset_num].ls() if str(i).find('.hdf5')!=-1]
    for f in data_options:
        fn = str(f).split('/')[-1]
        if fn.split('_')[0].isnumeric():
            data =f
    print(data)
    return data

def load_in(data_path):
    dp = hs.load(data_path)
    test_data = pxm.signals.ElectronDiffraction2D(dp)
    return test_data        
        
def ml_resize(dt):    
    chop1 = int(np.round((dt.shape[1]-256)/2,0))
    chop2 = int(np.round((dt.shape[2]-256)/2,0))
    if chop1 >= 1:
        dt = dt[:,chop1:-chop1,:]
    if chop2 >= 1:
        dt = dt[:,:,chop2:-chop2,]
    if dt.shape[1]%128 == 0:
        skip1 =  dt.shape[1]//128
        dt = dt[:,::skip1,:]
    if dt.shape[2]%128 == 0:
        skip2 =  dt.shape[2]//128
        dt = dt[:,:,::skip2]
    if dt.shape[1] * dt.shape[2] != 16384:
        print('Warning: resizing will be very slow')
        dt = resize(dt, (128,128))
    return dt

def get_label_overlap(x,y,map1,map2):
    return np.where(np.where(map1==x,1,0)*np.where(map2==y,1,0)==1,1,0)

def get_overlap_coords(x,y,map1,map2):
    return np.asarray(np.where(np.where(map1==x,1,0)*np.where(map2==y,1,0)==1))

def get_flat_overlap_coords(x,y,map1,map2):
    map1 = flatten_nav(map1)
    map2 = flatten_nav(map2)
    return np.asarray(np.where(np.where(map1==x,1,0)*np.where(map2==y,1,0)==1))[0]

def asap(shape):
    '''as square as possible'''
    for i in range(1,int(np.round(shape**0.5,0))+1):
        if shape % i == 0:
            rsize = i
    return rsize

def get_training_paths(root):
    p1= [x for x in root.ls()]

    p2 =[[x for x in each.ls() if (str(x).split('.')[-1] =='hdf5' or str(x).split('.')[-1] =='hspy')] for each in p1]

    p3 =[[x for x in each if str(x).split('/')[-1][0].isnumeric()] for each in p2]

    p4 =[[x for x in each if str(x).split('.')[-2][-1].isnumeric()] for each in p3]

    return [str(x[0]) for x in p4]


class MetaData():
    def __init__(self):
        self.crop = [[None,None],[None,None]]
        self.skips = None
    def current_timestamp(self):
        import time
        return ''.join([str(x) for x in time.localtime()[:5]])
    
#class Path(type(pathlib.Path())):
#    def ls(self):
#        return list(self.iterdir())

from stemutils.io import Path
    
class ModelInfo():
    def __init__(self):
        pass
    
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path as mplPath


class SelectFromCollection(object):
    """Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool highlights
    selected points by fading them out (i.e., reducing their alpha values).
    If your collection has alpha < 1, this tool will permanently alter them.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, ax2, img,bar, n_clusts, sample, live_update, alpha_other=1):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other
        
        if live_update == True:
        
            self.canvas2 = ax2.figure.canvas
            self.ax2=ax2

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, self.Npts).reshape(self.Npts, -1)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []
        self.bar = bar
        self.nclust = n_clusts
        self.sample=sample
        self.live_update = live_update
        
    def cmapper(self,x, xmax):
        i = ((np.cos(x/xmax *2*np.pi))+1)/2
        j = ((np.sin(x/xmax *2*np.pi))+1)/2
        return [i,j,0,1]

    def onselect(self, verts):
        path = mplPath(verts)
        self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
        #self.fc[:, -1] = self.alpha_other
        self.fc[self.ind] = self.cmapper(self.bar.widget.result, self.nclust)
        self.collection.set_facecolors(self.fc)
        
        if self.live_update == True:
        
            cs = np.unique(self.fc, axis=0)
            blank = np.zeros_like(self.fc[:,0])
            for i in range(cs.shape[0]):
                blank[np.where((self.fc==cs[i]).sum(axis=1)==4)[0]] += i+1
            print(blank)

            self.ax2.imshow(blank.reshape(self.sample.raw_data.data.shape[0:2]))
            self.canvas2.draw_idle()
            
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
    

class ProcessedSample():
    def __init__(self, dp, dataset, generic_path = True, lazy=True):
        '''dp should be the path to your data, dataset is a label used for folders 
        (mostly for saving VAE models and can be generally ignored)
        generic_path is also essentially useless, just leave it as True (is an artifact from an old data loading pipeline, will remove in future)
        '''
        if generic_path == True:
            self.time_stamp = None
        else:
            self.time_stamp=str(dp).split('/')[-1].split('_')[0]
        self.generic = generic_path
        self.data_path = dp
        self.raw_data = hs.load(self.data_path, lazy = lazy)
        self.dataset = dataset
        metadata = MetaData()
        metadata.time_stamp = self.time_stamp
        metadata.dataset = self.dataset
        self.metadata = metadata
        self.all_maps = {}
        self.all_patterns={}
        self.intermediate_map={}
        self.all_reps = {}
        self.ml_processed_data = {}
        
        
    ## Utility Section ###
    
    def save_all(self,save_path,new_folder=None):
        if new_folder!= None:
            sp = Path(f'{save_path}/{new_folder}')
            sp.mkdir()
        else:
            sp = Path(save_path)
        for key in self.all_maps.keys():
            val = self.all_maps[key]
            np.save(sp/f'all_maps__{key}__.npy', val)
        for key in self.all_patterns.keys():
            val = self.all_patterns[key]
            np.save(sp/f'all_patterns__{key}__.npy', val)
        for key in self.all_reps.keys():
            val = self.all_reps[key]
            np.save(sp/f'all_reps__{key}__.npy', val)
        adjust_nav_info = f'x_{self.metadata.crop[0][0]}-{self.metadata.crop[0][1]}_y_{self.metadata.crop[1][0]}-{self.metadata.crop[1][1]}_skips_{self.metadata.skips}_bin_{self.metadata.bin}'
        with open(sp/"scale.txt", "w") as text_file:
            text_file.write(adjust_nav_info)
            
    def load_all(self, load_path):
        lp = Path(load_path)
        for each in lp.ls():
            fp = str(each)
            f = fp.split('/')[-1]
            ft = f.split('.')[-1]
            if ft == 'txt':
                with open(fp,'r') as nav_info_f:
                    nav_info = nav_info_f.readline()
                nav_data = nav_info.split('_')
                if nav_data[1].split('-')[0] != 'None':       
                    crop = [[int(i) for i in nav_data[1].split('-')], [int(i) for i in nav_data[3].split('-')]]
                    self.adjust_nav(crop=crop)
                if nav_data[-3] != 'None':
                    skips = int(nav_data[-3])
                    self.adjust_nav(skips=skips)
                if nav_data[-1] != 'None':
                    nbin = int(nav_data[-1])
                    self.rebin(nbin)
            if ft == 'npy':
                dict_info = f.split('__')
                if dict_info[0] =='all_maps':
                    self.all_maps[dict_info[1]] = np.load(fp)
                if dict_info[0] =='all_patterns':
                    self.all_patterns[dict_info[1]] = np.load(fp)
                if dict_info[0] =='all_reps':
                    self.all_reps[dict_info[1]] = np.load(fp)

    
    def adjust_nav(self, crop= None, skips=None):
        '''Operations are performed inplace so take care when rerunning cells
        
        Crop operation occurs first and should take the form [[xmin, xmax],[ymin, ymax]]
        where xmin to xmax is left to right and ymin to ymax is top to bottom
        
        The skips will then apply to this cropped region'''
        if crop != None:
            self.raw_data = self.raw_data.inav[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
            self.metadata.crop = crop
        if skips != None:
            self.raw_data = self.raw_data.inav[::skips,::skips]
            self.metadata.skips = skips
            
    def rebin(self, nav_factor):
        if type(self.raw_data.data) != np.ndarray:
            sig = hs.signals.Signal2D(self.raw_data.data.compute())
        else:
            sig = hs.signals.Signal2D(self.raw_data.data)
        ns = (nav_factor, nav_factor,1, 1)
        self.raw_data = sig.rebin(None, ns)
        self.raw_data.data = self.raw_data.data/(nav_factor**2)
        self.metadata.bin = nav_factor
    
    def reset_nav(self):
        self.raw_data = hs.load(self.data_path, lazy = True)
    
    def get_patterns_from_map(self,tag):
        fig, self.all_patterns[tag] = show_patterns(flatten_nav(self.raw_data.data),self.all_maps[tag])
    
    def get_map_from_string(self, string):
        out_map = None
        if string == 'mi':
            try:
                out_map = self.all_maps['mi']
            except:
                print("Haven't evaluated a peakwise map")
        if string == 'vae':
            try:
                out_map = self.all_maps['vae']
            except:
                print("Haven't evaluated a VAE map")
        if string == 'pca':
            try:
                out_map = self.all_maps['pca']
            except:
                print("Haven't evaluated a VAE map")
        else:
            try:
                out_map = self.all_maps[string]
            except:
                print("Either haven't evaluated this map or don't recognise the tag")
        return out_map
    
    def workflow_help(self, workflow = None):
        '''Workflows are "mi", "pca" or "pca"'''
        if workflow == 'pca':
            print('Overview: 1. get_PCA_dimensionality(); 2. get_PCA_signal(); either 3. cluster_base_PCA() or 3. get_reduced_PCA() then get_reduced_PCA_clustering()')
        if workflow == 'vae':
            print('Overview: 1. load_model(); 2. encode(); 3. get_vae_map()')
        if workflow == 'mi':
            print('Overview: 1. find_all_peaks(); 2. run_mi(); 3. get_mi_map()')
    def imshow(self, imdata, tag=None, colorbar = True, cmap = 'turbo'):
        plt.figure()
        if tag ==None:
            plt.imshow(imdata, cmap= cmap)
            if colorbar == True:
                plt.colorbar()
        else:
            plt.imshow(self.all_maps[tag], cmap=cmap)
            if colorbar == True:
                plt.colorbar()
        
    def get_map(self, clust_func, rep, tag, store=True, output = False, shape = None, show=True):
        ''' clust func: either "gm" or "dbscan"
        if gm: func_param should be [number of components]
        if dbscan func_param should be [radius of interest, minimum samples]'''
        #if clust_func == 'gm':
        #    from sklearn.mixture import GaussianMixture as GM
        #    scaling_factor = rep[:,0].max() - rep[:,0].min() 
        #    c = GM(func_params[0]).fit_predict(rep/scaling_factor)+1
        #if clust_func == 'dbscan':
        #    from sklearn.cluster import DBSCAN as DB
        #    c = DB(eps = func_params[0], min_samples = func_params[1]).fit_predict(rep)+1
        c = clust_func.fit_predict(rep)+1
        if shape == None:
            raw_shape = self.raw_data.data.shape
        else:
            raw_shape= shape
        if self.generic != True:
            recover_locs = np.load(f'/dls/e02/data/2021/mg28034-1/processing/NumpyMLdata/{self.metadata.dataset}/{self.metadata.time_stamp}_locations.npy')
        else:
            if raw_shape[1]==1:
                recover_locs = np.arange(raw_shape[0])
            else:
                recover_locs = np.moveaxis(np.unravel_index(np.arange(raw_shape[0]*raw_shape[1]),(raw_shape[0],raw_shape[1])),0,1)
        blank = np.zeros((raw_shape[0],raw_shape[1]))
        for ind, coords in enumerate(recover_locs):
            if raw_shape[1]==1:
                blank[coords] += c[ind]
            else:
                blank[coords[0],coords[1]] += c[ind]
        if show == True:

            fig1= plt.figure()
            plt.imshow(blank,'turbo')
            plt.colorbar()
        if store == True:
            self.all_maps[tag] = blank
        if output == True:
            return c, blank
        
    def get_map_patterns(self,tag=None, quick = True, method='max', recompute= False):
        if tag!=None:
            clust_map = self.all_maps[tag]
        flat_raw = flatten_nav(self.raw_data.data)
        if recompute == False:
            if (tag in list(self.all_patterns.keys())) == True:
                fig = tile_stack(self.all_patterns[tag])
            else:
                fig, self.all_patterns[tag] = show_patterns(flat_raw, clust_map,method)
        else:    
            fig, self.all_patterns[tag] = show_patterns(flat_raw, clust_map,method)
        
    def cluster_rep_iterative(self, clust_func, func_params, rep, tag, index=None, provide_starting_map = None):
        if index == None:
            self.intermediate_map[tag] = {'primary':{}, 'subsequent':{}}
            if type(provide_starting_map) != type(None):
                self.intermediate_map[tag]['primary']['c'], self.intermediate_map[tag]['primary']['map'] = flatten_nav(provide_starting_map), provide_starting_map
            else:
                self.intermediate_map[tag]['primary']['c'], self.intermediate_map[tag]['primary']['map'] = self.get_map(clust_func, func_params, rep, tag, store=False, output = True, shape = None)
            plt.figure()
            plt.scatter(rep[:,0], rep[:,1],c=self.intermediate_map[tag]['primary']['c'],s=1)
            plt.colorbar()
            for each in range(self.intermediate_map[tag]['primary']['c'].max()+1):
                self.intermediate_map[tag]['subsequent'][each]={}
        else:
            if type(index)==int:
                inter_rep = rep[np.where(self.intermediate_map[tag]['primary']['c']==index)]
                self.intermediate_map[tag]['subsequent'][index]['rep'] = inter_rep
                self.intermediate_map[tag]['subsequent'][index]['c'], md = self.get_map(clust_func, func_params, inter_rep, tag, store=False, output = True, shape = (inter_rep.shape[0],1), show=False)
                plt.figure()
                plt.scatter(inter_rep[:,0], inter_rep[:,1], s=1, c = self.intermediate_map[tag]['subsequent'][index]['c'])
                plt.colorbar()
                
            if type(index)==list:
                positions = []
                prime_index = np.where(self.intermediate_map[tag]['primary']['c']==index[0])[0]
                positions.append(prime_index)
                inter_rep = rep[prime_index]
                for each in index[1:]:
                    sub_index = np.where(self.intermediate_map[tag]['primary']['c']==each)[0]
                    positions.append(sub_index)
                    inter_rep = np.concatenate((inter_rep, rep[sub_index]),axis=0)
                self.intermediate_map[tag]['subsequent'][index[0]]['rep'] = inter_rep
                self.intermediate_map[tag]['subsequent'][index[0]]['c'], md = self.get_map(clust_func, func_params, inter_rep, tag, store=False, output = True, shape = (inter_rep.shape[0],1), show=False)
                self.intermediate_map[tag]['subsequent'][index[0]]['combination_positions'] = positions
                plt.figure()
                plt.scatter(inter_rep[:,0], inter_rep[:,1], s=1, c = self.intermediate_map[tag]['subsequent'][index[0]]['c'])
                plt.colorbar()
                for each in index[1:]:
                    self.intermediate_map[tag]['subsequent'][each]['combination'] = index[0]
    def collapse_iterative_cluster(self,rep,tag):
        total = 0
        blank = np.zeros_like(self.intermediate_map[tag]['primary']['c'])
        for each in range(self.intermediate_map[tag]['primary']['c'].max()+1):
            if len(self.intermediate_map[tag]['subsequent'][each]) > 1:
                new = self.intermediate_map[tag]['subsequent'][each]['c']+total
                total+= self.intermediate_map[tag]['subsequent'][each]['c'].max()
                try:
                    positions = self.intermediate_map[tag]['subsequent'][each]['combination_positions']
                    runpos = 0
                    for pos in positions:
                        lenpos = len(pos)
                        blank[pos] = new[runpos:runpos+lenpos]
                        runpos += lenpos
                except:
                    blank[np.where(self.intermediate_map[tag]['primary']['c']==each)] = new
            elif len(self.intermediate_map[tag]['subsequent'][each]) == 1:
                pass
            else:
                total +=1
                blank[np.where(self.intermediate_map[tag]['primary']['c']==each)] = total
        self.all_maps[tag] = blank.reshape(self.intermediate_map[tag]['primary']['map'].shape)
        
    def manual_segment(self, rep, n_clusts, live_update = True, alpha=0.8):

        plt.ion()
        blank = np.zeros_like(rep[:,0])
        
        bar = interact(lambda x : x, x=(1,n_clusts,1))

        fig, ax = plt.subplots()
        
        if live_update == True:
        
            fig2, ax2 = plt.subplots()
            img = ax2.imshow(blank.reshape(self.raw_data.data.shape[0:2]))
            
        else:
            ax2, img = None, None

        pts = ax.scatter(rep[:, 0], rep[:, 1], s=1, alpha = alpha)
        selector = SelectFromCollection(ax, pts,ax2,img,bar,n_clusts,self, live_update)

        plt.show()

        return selector
    
    def collapse_manual_cluster(self, manual, tag):
        cs = np.unique(manual.fc, axis=0)
        blank = np.zeros_like(manual.fc[:,0])
        for i, col in enumerate(manual.fc):  
            blank[i] = int(np.where((col==cs).sum(axis=1)==4)[0]+1)
        self.all_maps[tag] = blank.reshape(self.raw_data.data.shape[0:2])
    
            

    
    ##PCA Section####

    def get_PCA_dimensionality(self, nav_skips = 4,nav_crop = 2):
        find_PCA_dims(self.raw_data, nav_skips, nav_crop)
    def get_PCA_signal(self, ndims, nav_skips = 4, nav_crop = 2, tag = None, decomp_thresh = 1e-1, norm= True, specific_data = None,  inv = False, use_mask = True, inc_nav = True):
        if np.all(specific_data == None):
            self.PCA_signal, self.PCA_decomp = get_PCA_signal(self.raw_data, self.raw_data, ndims, nav_skips, nav_crop, decomp_thresh, norm, inv, use_mask = use_mask, inc_nav = inc_nav)
        else:
            self.PCA_signal, self.PCA_decomp = get_PCA_signal(specific_data,self.raw_data, ndims, nav_skips, nav_crop, decomp_thresh, norm, inv, use_mask = use_mask, inc_nav = inc_nav)
        if tag != None:
            self.all_reps[tag] = self.PCA_signal
    def cluster_base_PCA(self, nclusts, tag = 'PCA_base'):
        
        gm_model = GM(nclusts).fit(flatten_nav(self.PCA_signal))
        #gm_model.
        self.PCA_full_clusts= GM(nclusts).fit_predict(flatten_nav(self.PCA_signal))
        self.PCA_full_map = self.PCA_full_clusts.reshape(self.raw_data.data.shape[:2])
        flat_raw = flatten_nav(self.raw_data.data)
        self.PCA_full_patterns_fig, self.PCA_full_patterns = show_patterns(flat_raw, self.PCA_full_clusts.reshape(self.raw_data.data.shape[0:2]))
        self.all_maps[tag] = self.PCA_full_clusts.reshape(self.raw_data.data.shape[0:2])
        self.all_patterns[tag] = self.PCA_full_patterns
        self.all_reps[tag] = flatten_nav(self.PCA_signal)
    def get_reduced_PCA(self, perplexity=1000, niters=500, lr = 200, ee=12, pre_sep=True, tag=None):
        self.PCA_rep, self.PCA_rep_clusts = PCA_reduced_representation(self.PCA_signal, perplexity, niters,lr,ee, pre_sep)
        if tag != None:
            self.all_reps[tag] = self.PCA_rep
    def plot_reduced_PCA(self):
        X,Y = np.moveaxis(self.PCA_rep, 0,1)
        plt.figure()
        plt.scatter(X,Y, s = 1, alpha = 0.1)
    def get_reduced_PCA_clustering(self, clust_func, func_params):
        '''So far the choice of functions are "gm" or "dbscan", the parameters associated should be 
        gm: [number of clusters] or dbscan: [search radius, minimum neighbours]'''
        clusts, pfig, patts= PCA_clustering(self.raw_data, self.PCA_rep, self.PCA_rep_clusts, clust_func, func_params)
        self.PCA_cluster_data = clusts
        self.PCA_cluster_pattern_data = patts
        self.PCA_cluster_patterns = pfig
        
    ##### VAE Section ####
    def set_model_data(self, model_basepath, model_type, latent_dims, bs=256, use_generic_model = True):
        self.metadata.model = ModelInfo()
        self.metadata.model.dims = latent_dims
        self.metadata.model.basepath = model_basepath
        self.metadata.model.type = model_type
        if use_generic_model == True:
            trained_set = 'Generic'
        else:
            trained_set = self.metadata.dataset
        if Path(f'{model_basepath}/{trained_set}').exists()!= True:
            Path(f'{model_basepath}/{trained_set}').mkdir()
        if Path(f'{model_basepath}/{trained_set}/{self.metadata.model.type}').exists()!= True:
            Path(f'{model_basepath}/{trained_set}/{self.metadata.model.type}').mkdir()
        self.metadata.model.path = Path(f'{model_basepath}/{trained_set}/{self.metadata.model.type}')
        self.metadata.model.bs = bs
    def pretrain_model(self, dps, epochs =100, nav_skip = 6):
        pretrain_model(dps, self.metadata, max_epochs = epochs, nav_skip=nav_skip)      
    def find_best_model(self, epochs=100, resize_latent = None, beta= 1):
        self.model = prepare_model(self.metadata, resize_latent, beta)
    def set_model(self, model):
        self.model = model
    def refine_model(self, epochs=100, use_full_ds = True, dmanip = None,lazy_load = True, input_data_tag = None):
        if input_data_tag != None:
            input_data = self.ml_processed_data[input_data_tag]
        else:
            input_data = None     
        self.model = transfer_learn(self.model,self.metadata, self.data_path, epochs, self.raw_data, use_full_ds, dmanip,lazy_load, input_data)
    def save_ml_manipulation(self,tag, dmanip, *args):
        dt = self.raw_data.data
        dt = flatten_nav(dt)
        self.ml_processed_data[tag] =  dmanip(dt, *args)


    def encode(self, tag = None, dmanip = None, input_data_tag = None, bn= 8):
        if input_data_tag ==None:
            if self.generic != True:
                np_data_path = f'/dls/e02/data/2021/mg28034-1/processing/NumpyMLdata/{self.metadata.dataset}/{self.metadata.time_stamp}_patterns.npy'
                dt = np.load(np_data_path)
            else:
                np_data_path = self.data_path
                dt = self.raw_data.data
                dt = flatten_nav(dt)
            if dmanip == None:
                if self.dmanip != None:
                    if self.ml_processed_data != None:
                        sample = self.ml_processed_data
                    else:
                        sample = self.dmanip(dt)      
                else:
                    sample=ml_resize(dt)
                    cb = np.where(np.mean(np.moveaxis(sample,0,-1),-1)>1,0,1)
                    sample *= cb.astype('uint8')
                    sample = sample.astype('float64')*(1/sample.max())
                    if self.metadata.model.type == 'cnn':
                        sample = sample[:,:,:,np.newaxis]
                    print(sample.shape)
            else:
                sample = dmanip(dt)
                print('data processed')
        else:
            sample = self.ml_processed_data[input_data_tag]
        self.vae_input = sample
        batch = np.array_split(sample, bn)
        encoded_data= self.model.encoder(batch[0])[2].numpy()
        for array in batch[1:]:
            encoded_data = np.concatenate((encoded_data, self.model.encoder(array)[2].numpy()), axis = 0)
        self.encoded_data = encoded_data
        if tag != None:
            self.all_reps[tag] = self.encoded_data

    def encode_multi(self, extra_data, tag = None, dmanip = None, input_data_tag = None, bn= None):
        sample = self.ml_processed_data[input_data_tag]
        self.vae_input = sample
        if bn == None:
                bn = asap(sample.shape[0])
        batch = np.split(sample, bn)
        batch_eds = np.split(extra_data, bn)
        encoded_data= self.model.encoder({'enc_input':batch[0],'eds_input':batch_eds[0]})[2].numpy()
        print('enc: ',self.model.encoder({'enc_input':batch[0],'eds_input':batch_eds[0]}))
        for array_i, array in enumerate(batch[1:]):
            array_eds = batch_eds[array_i]
            encoded_data = np.concatenate((encoded_data, self.model.encoder({'enc_input':array,'eds_input':array_eds})[2].numpy()), axis = 0)
        self.encoded_data = encoded_data
        if tag != None:
            self.all_reps[tag] = self.encoded_data
            
    def get_encoded_bic(self):
        bics = []
        for ncom in range(1,20):
            print(ncom)
            gmm = GM(n_components=ncom).fit(self.encoded_data)
            bics.append(gmm.bic(self.encoded_data))
        fig1 = plt.figure()
        plt.plot(range(1,20), bics)
        self.vae_bic = fig1
    
        
    def VAE_reduced_representation(self, perplexity, niters, lr=200, ee=12, tag = None):
        #can try to pre-seperate out the sample from the background, this can help with speeding up TSNE
        self.VAE_tsne = TSNE(n_components=2, perplexity=perplexity, verbose=2, n_iter=niters, learning_rate = lr, early_exaggeration= ee).fit_transform(self.encoded_data)
        if tag != None:
            self.all_reps[tag] = self.VAE_tsne
        
    def inspect_model(self, bn= 8):
        batch = np.array_split(self.encoded_data, bn)
        decoded_data= self.model.decoder(batch[0]).numpy()
        for array in batch[1:]:
            decoded_data = np.concatenate((decoded_data, self.model.decoder(array).numpy()), axis = 0)
        shape = list(self.raw_data.data.shape[:2])
        shape.append(128)
        shape.append(128)
        decoded_data = decoded_data.reshape(shape)
        self.decoded_data = decoded_data
        hs.signals.Signal2D(decoded_data).plot()
        
    def chart_terrain(self, dim1_info, dim2_info, plot = True):
        '''dim info should be a tuple of [dimension index, lower limit, upper limit, number of steps]'''
        example = self.encoded_data[10]
        blank = np.zeros((dim1_info[-1],dim2_info[-1],128,128))
        blank_preencoded = np.zeros((dim1_info[-1],dim2_info[-1],2))
        print(blank.shape)
        for xind, x in enumerate(np.linspace(dim1_info[1],dim1_info[2], dim1_info[-1])):
            for yind, y in enumerate(np.linspace(dim2_info[1],dim2_info[2], dim2_info[-1])):
                img = example.copy()
                blank_preencoded[xind,yind] = np.array((x,y))
                #img[dim1_info[0]] = x
                #img[dim2_info[0]] = y
                #nimg = self.model.decoder(img[np.newaxis,:]).numpy()
                #blank[xind,yind] = nimg[0,:,:,0]
        img = flatten_nav(blank_preencoded)
        n_batches = int(np.ceil(img.shape[0]//256))
        batches = [img[i*256:(i+1)*256] for i in range(n_batches+1)]
        nimg = [self.model.decoder(batch).numpy() for batch in batches]
        oimg = np.concatenate(nimg, axis = 0).reshape(blank.shape)
        self.terrain_signal = hs.signals.Signal2D(oimg)
        self.terrain_grid = (np.linspace(dim1_info[1],dim1_info[2], dim1_info[-1]), np.linspace(dim2_info[1],dim2_info[2], dim2_info[-1]))
        if plot == True:
            self.terrain_signal.plot()
            
    def chart_terrain_multi(self, dim1_info, dim2_info, plot = True):
        '''dim info should be a tuple of [dimension index, lower limit, upper limit, number of steps]'''
        example = self.encoded_data[10]
        blank = np.zeros((dim1_info[-1],dim2_info[-1],128,128))
        blank_preencoded = np.zeros((dim1_info[-1],dim2_info[-1],2))
        print(blank.shape)
        for xind, x in enumerate(np.linspace(dim1_info[1],dim1_info[2], dim1_info[-1])):
            for yind, y in enumerate(np.linspace(dim2_info[1],dim2_info[2], dim2_info[-1])):
                img = example.copy()
                blank_preencoded[xind,yind] = np.array((x,y))
                #img[dim1_info[0]] = x
                #img[dim2_info[0]] = y
                #nimg = self.model.decoder(img[np.newaxis,:]).numpy()
                #blank[xind,yind] = nimg[0,:,:,0]
        img = flatten_nav(blank_preencoded)
        n_batches = int(np.ceil(img.shape[0]//256))
        batches = [img[i*256:(i+1)*256] for i in range(n_batches+1)]
        nimg = [self.model.decoder(batch)[0].numpy() for batch in batches]
        oimg = np.concatenate(nimg, axis = 0).reshape(blank.shape)
        self.terrain_signal = hs.signals.Signal2D(oimg)
        self.terrain_grid = (np.linspace(dim1_info[1],dim1_info[2], dim1_info[-1]), np.linspace(dim2_info[1],dim2_info[2], dim2_info[-1]))
        if plot == True:
            self.terrain_signal.plot()

    def auto_cartography(self,tmp_dir, terr_resolution=10, n_comps = 10, n_segments=15, tag = 'vae_terrain_clustered', pad =0, pca_skips= 4, use_terr_decomp = True, inv = False, norm = False, mask_PCA = True, inc_nav = True):
        def full_ceil(val):
            return np.sign(val) *np.ceil(np.abs(val))

        terrxmax = full_ceil(self.encoded_data.T[0].max()) +pad
        terrymax = full_ceil(self.encoded_data.T[1].max()) +pad
        terrxmin = full_ceil(self.encoded_data.T[0].min()) -pad
        terrymin = full_ceil(self.encoded_data.T[1].min()) -pad
        yscale = int((terrymax-terrymin)*terr_resolution)
        xscale = int((terrxmax-terrxmin)*terr_resolution)     

        self.chart_terrain([0,terrxmin,terrxmax,xscale],[1,terrymin,terrymax,yscale])
        
        tmp_path = Path(f'{tmp_dir}/tmp.hspy')
        if tmp_path.exists():
            tmp_path.unlink()

        tersig = self.terrain_signal
        tersig = tersig.isig[:,:]*(64/tersig.data.max())
        tersig.change_dtype('uint8')
        tersig.save(f'{tmp_dir}/tmp.hspy', overwrite=True)

        terr = ProcessedSample(f'{tmp_dir}/tmp.hspy', 'terrain', generic_path = True)
        terr.extent = ((terrxmin, terrxmax),(terrymin,terrymax))
        
        if use_terr_decomp == True:
            terr.get_PCA_signal(n_comps,nav_skips=pca_skips,tag='pca',inv = inv, norm = norm, use_mask = mask_PCA, inc_nav= inc_nav)
            terr.cluster_base_PCA(n_segments)
        else:
            #terr.get_PCA_signal(n_comps,nav_skips=pca_skips,tag='pca', specific_data = self.raw_data)
            terr.get_PCA_signal(n_comps,nav_skips=pca_skips,tag='pca', specific_data = hs.signals.Signal2D(self.decoded_data),inv = inv, norm = norm, use_mask = mask_PCA, inc_nav= inc_nav)
            terr.cluster_base_PCA(n_segments)
            

        terr.imshow(None, 'PCA_base')

        terr_grid_x = np.linspace(terrxmin, terrxmax, xscale)[:,None]
        terr_grid_y = np.linspace(terrymin, terrymax, yscale)[None, :]

        terr_grid = np.repeat(terr_grid_x, yscale, axis = 1) + 1j*np.repeat(terr_grid_y, xscale, axis = 0)

        comp_encd = self.encoded_data[:,0] + 1j*self.encoded_data[:,1]

        terr_dgrid = np.abs(terr_grid[None, :,:] - comp_encd[:,None, None])

        arg_encd = np.asarray(np.unravel_index(np.argmin(terr_dgrid.reshape((terr_dgrid.shape[0], xscale*yscale)), axis = 1), (xscale, yscale))).T
        
        adjX,adjY = arg_encd.T
        
        self.terrain = terr

        self.all_maps[tag]=terr.all_maps['PCA_base'][adjX, adjY].reshape(self.raw_data.data.shape[:2])+1
            
    def latent_marked_terrain(self, imsize = [50,100], s = 1, c = 'red', alpha=1, m = 'o'):
        X, Y = self.encoded_data.copy().T
        self.chart_terrain([1,Y.max(),Y.min(),imsize[0]],[0,X.min(),X.max(),imsize[1]], plot = False)
        terr = self.terrain_signal.data
        terr = terr.mean(axis=-1).mean(axis=-1)

        terr = np.flip(terr, axis = 0)

        X -= X.min()
        X /= X.max()
        X *= (imsize[1] - 1)

        Y -= Y.min()
        Y /= Y.max()
        Y *= (imsize[0] -1)
        
        plt.figure()
        plt.imshow(terr)
        plt.scatter(X,Y, s = s, c = c, alpha = alpha, marker = m)
        plt.xlim([0,imsize[1] - 1])
        plt.ylim([0,imsize[0] -1])
        
    def distance_histogram(self, rep, index):
        from sklearn.neighbors import DistanceMetric
        dist = DistanceMetric.get_metric('l2')
        plt.figure()
        plt.hist(dist.pairwise(rep[::8])[index], 100)
        
    def get_decoded_pattern(self, patt1):   
        return self.model.decoder(patt1[np.newaxis,:]).numpy()[0,:,:,0]
    def explore_latent_terrain(self, plot_range = (-2.5,2.5,0.1)):
        from ipywidgets import interactive
        plt.figure(50)

        def f(a,b,c,d,e,f,g,h):
            lat=np.array((a,b,c,d,e,f,g,h))
            img = self.get_decoded_pattern(lat)
            plt.figure(50)
            plt.imshow(img)
            plt.show()

        interactive_plot = interactive(f, a=plot_range, b=plot_range,c=plot_range,d=plot_range,e=plot_range,f=plot_range,g=plot_range,h=plot_range,continuous_update= False)
        output = interactive_plot.children[-1]
        return interactive_plot

        
    #### MI Section ####    
    def find_all_peaks(self):
        self.peaks = find_all_peaks(self.raw_data)
    def run_mi(self, mutual_threshold = 0.05, without_mi = False, perplexity = 100, nruns = 500, lr= 0.1, ee=12, tag =None):
        self.mi_data, self.mi_signal, self.mi_figs = correlated_peaks_tsne(self.raw_data, self.peaks, self.metadata, mutual_threshold, without_mi, perplexity, nruns,lr,ee, verbose=True, show_figs=False, eg_ind=10)
        self.mi_rep = self.mi_figs[5]
        self.mi_bic = self.mi_figs[6]
        if tag != None:
            self.all_reps[f'{tag}_sig'] = self.mi_signal
            self.all_reps[f'{tag}_rep'] = self.mi_data
    def get_mi_map(self, clust_func, func_params, show_figs = False):
        '''So far the choice of functions are "gm" or "dbscan", the parameters associated should be 
        gm: [number of clusters] or dbscan: [search radius, minimum neighbours]'''
        if self.generic != True:
            recover_locs = np.load(f'/dls/e02/data/2021/mg28034-1/processing/NumpyMLdata/{self.metadata.dataset}/{self.metadata.time_stamp}_locations.npy')
        else:
            raw_shape = self.raw_data.data.shape
            recover_locs = np.moveaxis(np.unravel_index(np.arange(raw_shape[0]*raw_shape[1]),(raw_shape[0],raw_shape[1])),0,1)
        self.mi_maps, self.mi_map_data, self.mi_cluster_data, self.mi_cluster_probability = get_map_from_rep(self.mi_data, self.metadata, self.raw_data, recover_locs, clust_func, func_params)
        self.mi_clustered_rep, self.mi_cluster_map, self.mi_patterns = self.mi_maps
    def save_mi_map(self, map_type=None):
        dataset_root = f'/dls/e02/data/2021/mg28034-1/processing/Maps/{self.metadata.dataset}'
        if Path(dataset_root).exists()!= True:
            Path(dataset_root).mkdir()
        map_fileroot = f'{dataset_root}/{self.metadata.time_stamp}'
        if Path(map_fileroot).exists()!= True:
            Path(map_fileroot).mkdir()
        if map_type == 'confidence':
            data = self.mi_confidence_data
        if map_type == 'adjusted':
            data = self.mi_adjusted_data
        else:
            map_type = 'clusters'
            data = self.mi_cluster_data
        map_filepath = map_fileroot + f'/MI_{map_type}_{self.metadata.current_timestamp()}.npy'
        np.save(map_filepath, data)
        print(f'Saved to {map_filepath}')
    
        
    #### comparison    
    
    def compare_maps(self,tag1,tag2, quick=True):
        map1 = self.all_maps[tag1]
        map2 = self.all_maps[tag2]
        t1 = time.time()
        cm = clust.contingency_matrix(map1,map2)
        self.contingency_matrix = cm
        figlay= cm.shape

        region_label =0
        regions = np.zeros_like(map1)

        fig1,axs = plt.subplots(figlay[0], figlay[1], squeeze=False)
        overlap_data = []
        for x in range(figlay[0]):
            for y in range(figlay[1]):
                region_label +=1
                overlap = get_label_overlap(x+1,y+1,map1,map2)
                regions += (overlap*region_label)
                overlap_data.append(overlap)

                axs[x,y].imshow(overlap)
                axs[x,y].axis('off')
        self.map_overlaps = fig1
        self.map_regions = regions
        self.map_overlaps_data = np.asarray(overlap_data)
        cb = np.asarray(np.where(self.raw_data.data.mean(0).mean(0)>1,0,1))
        all_patts = None
        
        t2 = time.time()

        fig2,axs = plt.subplots(figlay[0], figlay[1], squeeze=False)
        flat_raw = np.asarray(flatten_nav(self.raw_data.data))
        t3 = time.time()
        for x in range(figlay[0]):
            for y in range(figlay[1]):
                coords = get_flat_overlap_coords(x+1,y+1,map1,map2)
                t4 = time.time()
                if coords.size>0:
                    temp_patts = flat_raw[coords]
                    print(temp_patts.shape, coords.shape)
                    patts = np.max(temp_patts, axis=0)*cb
                else:
                    patts = np.zeros_like(flat_raw[1])
                t5 = time.time()
                if type(all_patts) == type(None):
                    all_patts = patts[np.newaxis,:,:]
                else:
                    all_patts = np.concatenate((all_patts, patts[np.newaxis,:,:]), axis = 0)
                t55 = time.time()
                if quick == True:
                    trim_patt = (patts[30:-30, 30:-30])[::4,::4]
                else:
                    trim_patt = patts
                trim_patt = trim_patt.astype('uint8')

                axs[x,y].imshow(trim_patt)
                axs[x,y].axis('off')
                t6 = time.time()
        tl = time.time()
        self.overlap_patterns = fig2
        self.region_patterns = all_patts
        print(t55-t5, t6-t55, tl-t3, tl-t1)
        
    
    def refine_maps(self, nclust):
        self.tape = {}
        self.nclust = nclust
        cm_shape = self.contingency_matrix.shape

        self.compare_fig = plt.figure(figsize=(8,8))

        for i in range(self.contingency_matrix.size):
            self.tape[i] = []
            plt1 = plt.subplot(cm_shape[0],cm_shape[1],i+1)
            plt.imshow(self.region_patterns[i,50:-50,50:-50],cmap = 'turbo')
            plt.xticks([])
            plt.yticks([])
            plt1.set_picker(True)
            
        def onpick(event):
            plot_num = int(event.artist.get_geometry()[-1])
            data_num = plot_num -1
            fig = self.compare_fig
            plt1 = plt.subplot(cm_shape[0],cm_shape[1],plot_num)
            cmap = get_cmap('turbo')


            val = self.bar.widget.result/ 6
            rgb = cmap(val)[:-1]


            [plt1.spines[x].set_color(rgb) for x in ["bottom","top","left","right"]]
            [plt1.spines[x].set_linewidth(5) for x in ["bottom","top","left","right"]]

            self.tape[data_num].append(self.bar.widget.result)
        



        self.compare_fig.canvas.mpl_connect('pick_event', onpick)


        self.bar = interact(f, x=(0,nclust,1))
        
        
        
    def plot_refined_maps(self, tag):
        info = {}
        for x in range(self.nclust+1):
            info[x] = []
        for key,val in self.tape.items():
            try:
                info[val[-1]].append(key)
            except:
                continue


        blank_map = np.zeros(self.raw_data.data.shape[0:2])
        blank_regions = []
        region_patterns = []
        for key in info.keys():
            region = []
            pattern = []
            print(info[key])
            for each in info[key]:
                region.append(self.map_overlaps_data[each])
                if type(pattern) == list:
                    pattern = self.region_patterns[each][np.newaxis,:,:]
                else:
                    pattern=np.concatenate((pattern,self.region_patterns[each][np.newaxis,:,:]),axis = 0)
            region = np.asarray(region)
            print('made it here')
            blank_map += np.where(region.sum(axis=0)>=1,key,0)
            try:
                blank_regions.append(np.where(region.sum(axis=0)>=1,1,0))
            except:
                blank_regions.append([])
            try:
                region_patterns.append(pattern.max(0))
            except:
                region_patterns.append([])

        refined_map = plt.figure()
        plt.imshow(blank_map, cmap = 'turbo')

        fig2,ax = plt.subplots(self.nclust+1)
        for ind, patts in enumerate(region_patterns):
            if type(patts) == list:
                patts = np.zeros(self.raw_data.data.shape[2:])
            ax[ind].imshow(patts)
            
        self.all_maps[tag] = np.asarray(blank_map)+1
        self.all_patterns[tag] = patts
            
        self.refined_map = refined_map
        self.refined_patterns = fig2
        self.refined_map_data = blank_map
        self.refined_patterns_data = patts
        
    def combine_map_clusters(self,tag1, nclust):
        map1 = self.all_maps[tag1]
        self.temp_reduction_tag = tag1
        figlay= [map1.max()//asap(map1.max()),asap(map1.max())] 
        self.get_map_patterns(tag1)
        self.combine_patts = self.all_patterns[tag1]
        patts = self.combine_patts
        fig_shaped_patts = patts.reshape([figlay[0],figlay[1],patts.shape[1],patts.shape[2]])


        self.tape = {}
        self.nclust = nclust

        self.compare_fig = plt.figure(figsize=(8,8))

        for i in range(patts.shape[0]):
            self.tape[i+1] = [None]
            plt1 = plt.subplot(figlay[0],figlay[1],i+1)
            plt.imshow(patts[i,50:-50,50:-50],cmap = 'turbo')
            plt.xticks([])
            plt.yticks([])
            plt1.set_picker(True)

        def onpick(event):
            plot_num = int(event.artist.get_geometry()[-1])
            data_num = plot_num
            fig = self.compare_fig
            plt1 = plt.subplot(figlay[0],figlay[1],plot_num)
            cmap = get_cmap('turbo')


            val = self.bar.widget.result/ self.nclust
            rgb = cmap(val)[:-1]


            [plt1.spines[x].set_color(rgb) for x in ["bottom","top","left","right"]]
            [plt1.spines[x].set_linewidth(5) for x in ["bottom","top","left","right"]]

            self.tape[data_num].append(self.bar.widget.result)




        self.compare_fig.canvas.mpl_connect('pick_event', onpick)


        self.bar = interact(f, x=(0,nclust,1))

    def collapse_combined_map(self, tag):
        intag = self.temp_reduction_tag
        inmap = self.all_maps[intag]
        outmap = np.zeros_like(inmap)
        info = {}
        running_count = 1
        previously_seen = {}
        for key in self.tape.keys():
            if self.tape[key][-1] == None:
                cluster_value = running_count
                running_count += 1
            elif previously_seen.get(self.tape[key][-1]) == None:
                cluster_value = running_count
                previously_seen[self.tape[key][-1]] = cluster_value
                running_count += 1
            else:
                cluster_value = previously_seen[self.tape[key][-1]]
            outmap[np.where(inmap == key)] = cluster_value
        self.all_maps[tag] = outmap

    def save_map(self, tag, save_path):
        dataset_root = f'{save_path}{self.metadata.dataset}'
        if Path(dataset_root).exists()!= True:
            Path(dataset_root).mkdir()
        map_fileroot = f'{dataset_root}/{self.metadata.time_stamp}'
        if Path(map_fileroot).exists()!= True:
            Path(map_fileroot).mkdir()
        data = self.all_maps[tag]
        map_filepath = map_fileroot + f'/{tag}_{self.metadata.current_timestamp()}.npy'
        np.save(map_filepath, data)
        print(f'Saved to {map_filepath}')

    def look_for_map(self, tag, path):
        dataset_root = f'{save_path}{self.metadata.dataset}'
        map_filepath = map_fileroot + f'/{tag}_{self.metadata.current_timestamp()}.npy'
        print(Path(map_filepath).ls())
        
    def merge_labels(self, old_tag,ind_list, new_tag = None):
        in_map = self.all_maps[old_tag]
        for ind in ind_list[1:]:
            in_map = np.where(in_map == ind, ind_list[0], in_map)
        unique_inds = np.unique(in_map)
        out_map = np.zeros_like(in_map)
        for x, ui in enumerate(unique_inds):
            out_map[np.where(in_map == ui)] = x +1 
        if new_tag == None:
            self.all_maps[old_tag] = out_map
        else:
            self.all_maps[new_tag] = out_map
        return out_map


#load some packages in
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import random as python_random
from numba import njit
from tensorboard.plugins.hparams import api as hp
from stemutils.io import Path
import hyperspy.api as hs
import concurrent.futures
from skimage.transform import resize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from functools import lru_cache
import json
import itertools

#set some variables
print('Using TensorFlow v%s' % tf.__version__)
plt.style.use('default')
python_random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)


#define some functions

###################################################
########### Data Preprocessing ####################
###################################################

def batch_resize(d, bs=512):
    if len(d.shape) == 4:
        flat_d = flatten_nav(d)
    else:
        flat_d = d
    n_batches = int(np.ceil(flat_d.shape[0]//bs))
    batches = [flat_d[i*bs:(i+1)*bs] for i in range(n_batches+1)]
    if len(batches[-1])==0:
        batches.pop(-1)
    print(len(batches[-1]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as exe:
        res = [exe.submit(resize, batch, (batch.shape[0],128,128)) for batch in batches]
    r_batches = [f.result() for f in res]
    return np.concatenate(r_batches, axis = 0).reshape((flat_d.shape[0],128,128))

def data_manip(d, bs = 512):
    if type(d) != np.ndarray:
        print('dask to numpy')
        d = d.compute()
        print('dask to numpy done')
    print('started data manipulations')
    #d = resize(d,(d.shape[0],128,128))
    print('resized')
    d = d.astype('float32')
    for i in range(d.shape[0]):
        d_max = np.max(d[i])
        if d_max <= 0:
            d_max = 1
        d[i] = d[i]/d_max
    d = batch_resize(d, bs)
    scaler = np.log(1001)
    return np.log((d*1000)+1)/scaler 

def data_manip_lowq(d, central_box = 128):
    pxc, pyc = d.shape[1]//2, d.shape[2]//2 
    pxl, pxu = pxc - central_box//2, pxc + central_box//2 
    pyl, pyu = pyc - central_box//2, pyc + central_box//2 
    
    d = d[:, pxl:pxu, pyl:pyu]
    if type(d) != np.ndarray:
        print('dask to numpy')
        d = d.compute()
        print('dask to numpy done')
    print('started data manipulations')
    #d = resize(d,(d.shape[0],128,128))
    print('resized')
    d = d.astype('float32')
    for i in range(d.shape[0]):
        d_max = np.max(d[i])
        d[i] = d[i]/d_max
    
    scaler = np.log(1001)
    return np.log((d*1000)+1)/scaler 



###################################################
###################################################
###################################################

def flatten_nav(sig):
    shape = [sig.shape[0]*sig.shape[1]]
    for i in sig.shape[2:]:
        shape.append(i)
    return sig.reshape(shape)


class My_Custom_Generator(keras.utils.Sequence) :
    def __init__(self, image_filenames,  batch_size) :
        self.image_filenames = image_filenames
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
    
    
    @lru_cache(None)
    def __getitem__(self, idx) :
        batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
        out_img = np.asarray([np.load(file_name)[:,:,None] for file_name in batch_x])
        return out_img, out_img
        #return batch_x, batch_y
        
        
class Array_Generator(keras.utils.Sequence) :
    def __init__(self, images,  batch_size) :
        self.images = images
        self.batch_size = batch_size

    def __len__(self) :
        return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
    
    
    @lru_cache(None)
    def __getitem__(self, idx) :
        out_img = self.images[idx * self.batch_size : (idx+1) * self.batch_size, :,:,None]
        return out_img, out_img
        #return batch_x, batch_y

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    



def custom_loss(x,y):
    n_img = 128
    return tf.reduce_mean(keras.losses.binary_crossentropy(x, y)) * n_img * n_img

def remove_background(sample, thresh = 500, old_tag=None, new_tag=None,blanker = 30):
    d = sample.raw_data.data.copy()
    d_shape = d.shape
    n_shape, p_shape = d_shape[0:2], d_shape[2:]
    ps0 = p_shape[0] //2
    try:
        d[:,:,ps0- blanker:ps0 + blanker, ps0 - blanker: ps0 + blanker] = np.zeros((2*blanker,2*blanker))
    except:
        d = d.compute()
        d[:,:,ps0- blanker:ps0 + blanker, ps0 - blanker: ps0 + blanker] = np.zeros((2*blanker,2*blanker))
    maskx, masky = np.where(d.sum(axis=(2,3))<thresh)
    if old_tag !=None:
        clustmap = sample.all_maps[old_tag].copy()
        clustmap += 1
        clustmap[maskx, masky] = 0 
        newmap = np.zeros_like(clustmap)
        for i, o in enumerate(np.unique(clustmap)):
            newmap[np.where(clustmap == o)] = i
        newmap += 1
        if new_tag != None:
            sample.all_maps[new_tag] = newmap
        return newmap
    else:
        return np.where(d.sum(axis=(2,3))<thresh, 0, 1)

def show_cluster_patterns(sample, tag):
    uis = np.unique(sample.all_maps[tag])
    od = np.zeros((uis.size, 512, 256))
    for x,i in enumerate(uis):
        p = resize(sample.all_patterns[tag][x], (256,256))
        n = resize(np.where(sample.all_maps[tag] == i, p.max(), 1), (256,256))
        p[0,:] = p.max()
        o = np.concatenate([n,p], axis = 0)
        od[x] = o
    return hs.signals.Signal2D(od)

def signal_boosted_scan(sample, tag):
    uts = np.unique(sample.all_maps[tag])
    blank = np.zeros_like((sample.raw_data))
    blank = blank.astype('float32')
    for i,t in enumerate(uts):
        print(i,t)
        blank[np.where(sample.all_maps[tag]==t)] = sample.all_patterns[tag][i]
    return hs.signals.Signal2D(blank)

def inv_sbs(sample, sbs, tag = 'vl_vae', sp = (0,0), return_fig = False, interactive = True, **kwargs):
    sbsg = np.repeat(sbs.data.sum(axis= (2,3))[:,:,None],3, -1)
    sbsg /= sbsg.max()
    
    def boost(array):
        return np.log10(np.log10(array+1)+1)

    def format_ax():
        ax[0].set_frame_on(False)
        #ax[1].set_frame_on(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]}, figsize=(8,8))
    
    
    clust = sample.all_maps[tag][sp[0],sp[1]]

    clust_loc = np.where(sample.all_maps[tag] == clust)

    new_nav = sbsg.copy()

    new_nav[clust_loc] = np.array([0.1254902 , 0.69803922, 0.66666667])
    
    
    ax[0].imshow(new_nav)
    ax[1].imshow(boost(sbs.data[sp[0],sp[1]]), cmap= 'gray', **kwargs)

    format_ax()
    
    if interactive == True:
    
        global coords
        coords = []

        def onclick(event):
            global ix, iy
            ix, iy = np.round(event.xdata,0), np.round(event.ydata,0)
            print(ix, iy)

            coords.append((ix, iy))

            ax[0].clear()
            ax[1].clear()

            clust = sample.all_maps[tag][int(iy),int(ix)]

            clust_loc = np.where(sample.all_maps[tag] == clust)

            new_nav = sbsg.copy()

            new_nav[clust_loc] = np.array([0.1254902 , 0.69803922, 0.66666667])



            ax[0].imshow(new_nav)
            ax[1].imshow(boost(sbs.data[int(iy),int(ix)]), cmap = 'gray', **kwargs)

            format_ax()

            ax[0].draw()
            ax[1].draw()


            return coords

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

    if return_fig == True:
        return fig

from skimage.metrics import structural_similarity as SSI
from skimage.transform import PiecewiseAffineTransform, warp
from sklearn.neighbors import NearestNeighbors as kNN

def get_latgrid(sample, res=100):
    xmin, xmax = np.floor(np.min(sample.encoded_data[:,0])), np.ceil(np.max(sample.encoded_data[:,0]))
    ymin, ymax = np.floor(np.min(sample.encoded_data[:,1])), np.ceil(np.max(sample.encoded_data[:,1]))

    latgrid_res = res

    xgrid, ygrid = np.repeat(np.linspace(xmin, xmax, latgrid_res)[:,None],latgrid_res, axis = 1), np.repeat(np.linspace(ymin, ymax, latgrid_res)[None,:],latgrid_res, axis = 0)

    return np.concatenate([xgrid[:,:,None], ygrid[:,:,None]],axis = 2)

def get_latgrid_free(sample, xmin, xmax, ymin, ymax, res=100):
    latgrid_res = res

    xgrid, ygrid = np.repeat(np.linspace(xmin, xmax, latgrid_res)[:,None],latgrid_res, axis = 1), np.repeat(np.linspace(ymin, ymax, latgrid_res)[None,:],latgrid_res, axis = 0)

    return np.concatenate([xgrid[:,:,None], ygrid[:,:,None]],axis = 2)








def batch_calc_grad(img, radial_kernel, decoded_data, weighting_func, bs=256):
    
    ssi_ff = []
    n_batches = int(np.ceil(img.shape[0]//bs))
    batches = [img[i*bs:(i+1)*bs] for i in range(n_batches+1)]
    dec_batches = [decoded_data[i*bs:(i+1)*bs] for i in range(n_batches+1)]
    rs_batches = [b.reshape(b.shape[0]*b.shape[1]) for b in batches]
    cart_rs_batches = [np.concatenate([b.real[:,None], b.imag[:,None]], axis = 1) for b in rs_batches]
    for i, batch in enumerate(cart_rs_batches):
        t1 = time.time()
        print(i, n_batches)
        nimg = sample.model.decoder(batch).numpy()
        rs_nimg = nimg.reshape((int(nimg.size/(img.shape[1]*128*128)), img.shape[1], 128, 128))
        comp_patterns = dec_batches[i]
        for x, dec_pat in enumerate(comp_patterns):
            grad_ssi = np.asarray([weighting_func(dec_pat, y) for y in rs_nimg[x]])
            ssi_ff.append(np.sum(grad_ssi*radial_kernel))
        print(time.time()-t1, 'single thread')
    return np.asarray(ssi_ff)


def SSI_weighting(img1, img2):
    return 100*SSI(img1,img2)

def get_mobile_points(nn_comp_enc,steps, prev_mp_locs = (), thresh = 'mean', relative_locs = False):
    if thresh == 'mean':
        thresh = np.mean(np.abs(steps))
    if thresh == 'ten':
        thresh = np.max(np.abs(steps))/10
        print(thresh)
        print(np.where(np.abs(steps) > thresh))
    mp_locs = np.where(np.abs(steps) > thresh)
    mobile_points = nn_comp_enc[mp_locs]
    
    if len(prev_mp_locs) != 0:
        n_mp_locs = prev_mp_locs[mp_locs]
        
    if relative_locs == True:

        return mobile_points, n_mp_locs, mp_locs
    else:
        return mobile_points, n_mp_locs

def get_grad_and_decode_data(mobile_points, radial_kernel, r_scale_kernel = False, nn_scale = False):
    if r_scale_kernel ==False:
        grad_points = np.repeat(mobile_points[:,None], radial_kernel.shape[0], axis = 1) + radial_kernel[None, :]
    else:
        if nn_scale == False:
            rf = np.round((np.abs(mobile_points)/np.abs(mobile_points).min()),0).astype('int')
            grad_points = np.repeat(mobile_points[:,None], radial_kernel.shape[0], axis = 1) + r_scale_kernel*rf[:,None]*radial_kernel[None, :]
        else:
            sample_locs = np.concatenate((mobile_points.real[:,None], mobile_points.imag[:,None]), axis = 0)
            nbrs = kNN(n_neighbors=1, algorithm='ball_tree').fit(sample_locs)
            p_sep, indices = nbrs.kneighbors(sample_locs)
            closest = p_sep.min()
            norm_sep = p_sep/closest
            grad_points = np.repeat(mobile_points[:,None], radial_kernel.shape[0], axis = 1) + r_scale_kernel*norm_sep[:,None]*radial_kernel[None, :]

    dec_dat = get_terr_patts(np.concatenate([mobile_points.real[:,None], mobile_points.imag[:,None]],axis = 1))
    return grad_points, dec_dat

def sig_step_from_grad(d_gp, gradient_step, sigz=0.25, sigf=100):
    grad_mag = np.abs(d_gp)

    return sigmoid(grad_mag, sigz, sigf)*gradient_step*(d_gp/grad_mag) 

def norm_step_from_grad(d_gp, factor):
    grad_mag = np.max(np.abs(d_gp))
    
    return (d_gp/grad_mag)*factor 

def sigmoid(z, sigz=0.25, sigf=100):
    x = sigf*(z - sigz)
    return np.exp(-np.logaddexp(0, -x))


def adjust_encoding(mobile_points, grads, comp_enc, mp_locs):
    X,Y  = mobile_points.real, mobile_points.imag

    dX, dY = grads.real, grads.imag
    U, V = X+dX, Y+dY

    moved_points = U+1j*V

    migrated_points = comp_enc.copy()

    migrated_points[mp_locs] = moved_points
    
    return (X,Y), (U,V), migrated_points

def get_terr_patts(sample, img, bs =256, multi = False):
    n_batches = int(np.ceil(img.shape[0]//bs))
    batches = [img[i*bs:(i+1)*bs] for i in range(n_batches+1)]
    if multi == False:
    	nimg = [sample.model.decoder(batch).numpy() for batch in batches]
    if multi == True:
    	nimg = [sample.model.decoder(batch)[0].numpy() for batch in batches]
    return np.concatenate(nimg, axis = 0).reshape((img.shape[0], 128,128))


def sig_step_from_grad(d_gp, gradient_step, sigz=0.25, sigf=100):
    grad_mag = np.abs(d_gp)

    return sigmoid(grad_mag, sigz, sigf)*gradient_step*(d_gp/grad_mag) 

def sigmoid(z, sigz=0.25, sigf=100):
    x = sigf*(z - sigz)
    return np.exp(-np.logaddexp(0, -x))

def lin_thresh_step(d_gp, thresh, mag = 1):
    scale = np.abs(d_gp)
    return (np.where(scale>thresh, thresh, scale)/thresh)*(d_gp/scale)*mag

def scaled_thresh_step(d_gp, thresh, mobile_points, mag):
    sample_locs = np.concatenate((mobile_points.real[:,None], mobile_points.imag[:,None]), axis = 1)
    nbrs = kNN(n_neighbors=5, algorithm='ball_tree').fit(sample_locs)
    p_sep, indices = nbrs.kneighbors(sample_locs, n_neighbors = 2)
    print(p_sep.shape, p_sep[:,0])
    p_sep = p_sep[:,1]
    closest = p_sep.min()
    norm_sep = p_sep/closest
    
    scale = np.abs(d_gp)
    return (np.where(scale>thresh, thresh, scale)/thresh)*(d_gp/scale)*norm_sep*mag

def sig_step_from_grad(d_gp, gradient_step, sigz=0.25, sigf=100):
    grad_mag = np.abs(d_gp)

    return sigmoid(grad_mag, sigz, sigf)*gradient_step*(d_gp/grad_mag) 

def sigmoid(z, sigz=0.25, sigf=100):
    x = sigf*(z - sigz)
    return np.exp(-np.logaddexp(0, -x))

def lin_thresh_step(d_gp, thresh, mag = 1):
    scale = np.abs(d_gp)
    return (np.where(scale>thresh, thresh, scale)/thresh)*(d_gp/scale)*mag

def scaled_thresh_step(d_gp, thresh, mobile_points, mag):
    sample_locs = np.concatenate((mobile_points.real[:,None], mobile_points.imag[:,None]), axis = 1)
    nbrs = kNN(n_neighbors=5, algorithm='ball_tree').fit(sample_locs)
    p_sep, indices = nbrs.kneighbors(sample_locs, n_neighbors = 2)
    print(p_sep.shape, p_sep[:,0])
    p_sep = p_sep[:,1]
    closest = p_sep.min()
    norm_sep = p_sep/closest
    
    scale = np.abs(d_gp)
    return (np.where(scale>thresh, thresh, scale)/thresh)*(d_gp/scale)*norm_sep*mag


from sklearn.neighbors import KernelDensity
def get_density_net(sample, n_samples, n_bkg_samples, density_approx = 10,  bandwidth=0.5):
    D = sample.encoded_data.copy()
    np.random.shuffle(D)
    D = D[::density_approx]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(D)
    R = kde.sample(n_samples)
    
    
    
    xmin, xmax = np.floor(np.min(sample.encoded_data[:,0])), np.ceil(np.max(sample.encoded_data[:,0]))
    ymin, ymax = np.floor(np.min(sample.encoded_data[:,1])), np.ceil(np.max(sample.encoded_data[:,1]))
    
    print(xmin, xmax,ymin,ymax)
    s_samples = np.random.random((n_bkg_samples, 2))
    
    s_samples[:,0] *= np.abs((xmax - xmin))
    s_samples[:,1] *= np.abs((ymax - ymin))
    s_samples = s_samples + np.array((xmin, ymin))
    
    return np.concatenate((R, s_samples), axis = 0)

import sklearn.metrics.cluster as cmet

def get_map_label_df(map1):
    return np.asarray([np.where(map1 == uinds, 1, 0) for uinds in np.unique(map1)])

def get_cluster_label_overlap(map_pair):
    db1_df,db2_df = map_pair
    label_overlap = np.zeros((db1_df.shape[0], db2_df.shape[0]))
    for i, idf in enumerate(db1_df):
        for j, jdf in enumerate(db2_df):
            label_overlap[i,j] = np.sum(db1_df[i] * db2_df[j])/ np.sum(db1_df[i])
    return label_overlap

def find_map_label(pos, map1):
    return map1[pos]

def get_confidence_from_maps(maps):
    dfs = [x for x in map(get_map_label_df, maps)]

    cluster_overlaps = [x for x in map(get_cluster_label_overlap, [x for x in itertools.permutations(dfs, 2)])]

    overlap_inds = [x for x in itertools.permutations(np.arange(len(maps)), 2)]

    overlap_inds

    len(cluster_overlaps)

    map1 = maps[0]

    confidence = np.zeros_like(map1, dtype='float32')
    for point in range(len(map1)):
        labels = [i for i in map(find_map_label, np.repeat(point, len(maps)) , maps)]
        total = 0
        for cind, oinds in enumerate(overlap_inds):
            l1, l2 = labels[oinds[0]], labels[oinds[1]]
            total+=cluster_overlaps[cind][l1, l2]
        mean = total/len(overlap_inds)
        confidence[point] = mean
    return confidence



def refine_based_on_density(sample, density_cutoff = -9, n_bulk_samples = 500, sample_grid_res = 200,
                            gn = 0.1,density_approx = 5, bw = 0.4, n_sample_points = 2500, n_bkg_points = 500, 
                            show_net = True, rand_gradient_step = 0.01,rand_n_rsteps = 12, step_scale = 0.01, 
                            step_thresh = 0.001,show_step_size = True, show_net_movement= True, 
                            show_first_refinement = False,n_refine_steps = 200, show_refinement = True, 
                            animate_refinement = True):
    
    '''
    sample: ProcessedSample with the encoded data
    density_cutoff: Cutoff value below which the denisty gradient is ignored (default - -9)
    n_bulk_samples: number of samples to take from the density distribution (default - 500)
    sample_grid_res: The number of grid points to use to approximate the density gradient function (default - 200)
    gn: Gaussian noise maximum magnitude to be added to the randomly sampled density gradient points (default - 0.1)
    density_approx: the number of skips to take of randomly shuffled encoded data to approximate the density (default - 10)
    bw: bandwidth for the density approximation (default - 0.3)
    n_sample_points: number of net points to sample from the density gradient distribution (default - 4000)
    n_bkg_points: number of net points to sample uniformly (default - 1000)
    show_net: Show the positions of the net points (default - True)
    rand_gradient_step: The distance around net points to sample for gradient approximation (default - 0.01)
    rand_n_rsteps: The number of radial samples around net points to sample for grad approx (default - 12)
    step_scale: The magnitude of the largest steps you want in net movement (default - 0.01)
    step_thresh: The gradient cut-off, above which, the step size == step_scale (default - 0.001)
    show_step_size: Show a graph of the distribution of step sizes (default - True)
    show_net_movement: Show how the net points have distorted (default - True)
    show_first_refinement: Show how the sample points have moved after one step (default - True)
    n_refine_steps: Number of refinement steps to run (default - 500)
    show_refinement: Show how the sample points have moved after n steps (default - True)
    animate_refinement: Show an animation of the refinement process (default - True)
    '''
    
    #create a dictionary to hold some data that might be useful to return
    accessory_dict = {}
    
    #get a density based net
    #R = get_density_net(sample, n_sample_points, n_bkg_points, density_approx, bw)
    R = get_density_gradient_net(sample, n_sample_points, density_cutoff, n_bkg_points,n_bulk_samples, density_approx, sample_grid_res,bw, gn)
    
    accessory_dict['net'] = R
    
    #view the point distribution
    if show_net == True:
        plt.figure()
        plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10)
        plt.scatter(R[:,0], R[:,1], s = 20)
    
    #get all the sample points for the gradient 
    rand_latspace = R[:,0] + 1j*R[:,1]
    rand_radial_kernel = rand_gradient_step* np.exp(1j*np.pi*(np.linspace(0, 360, rand_n_rsteps+1)/180))[1:]
    rand_grad_p1, rand_decdat1 = get_grad_and_decode_data(rand_latspace, rand_radial_kernel)
    
    #calculate the gradients
    rand_delta_gp1 = batch_calc_grad(rand_grad_p1, rand_radial_kernel, rand_decdat1, SSI_weighting, 256)
    accessory_dict['grads'] = rand_delta_gp1
    
    #scale the gradients for step sizes
    rand_linsteps = lin_thresh_step(rand_delta_gp1, step_thresh, step_scale)
    if show_step_size == True:
        plt.figure()
        plt.plot(np.sort(np.abs(rand_linsteps)))
    accessory_dict['steps'] = rand_linsteps
    
    #adjust the net points
    rand_op1, rand_np1, rand_current_ps1= adjust_encoding(rand_latspace, rand_linsteps, rand_latspace, np.where(rand_latspace != None))
    accessory_dict['net_displacement'] = rand_np1
    if show_net_movement == True:
        plt.figure(figsize = (8,8))
        plt.scatter(rand_op1[0], rand_op1[1],s =20)
        plt.scatter(rand_np1[0], rand_np1[1],s =20)
        plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.2)
    #refine the points once
    test_data = sample.encoded_data.copy()

    rand_o_latspacer = np.asarray(rand_op1).T
    rand_n_latspacer = np.asarray(rand_np1).T

    rand_tform2 = PiecewiseAffineTransform()
    accessory_dict['transform'] = rand_tform2
    rand_tform2.estimate(rand_n_latspacer, rand_o_latspacer)

    rand_out_data2 = rand_tform2.inverse(test_data)
    
    accessory_dict['first_refinement'] = rand_out_data2.copy()

    if show_first_refinement == True:
        plt.figure()
        plt.scatter(test_data[:,0], test_data[:,1], s =10)
        plt.scatter(rand_out_data2[:,0], rand_out_data2[:,1], s =10)
        
    rand_refine_steps2=[]

    for i in range(n_refine_steps):
        rand_out_data2 = rand_tform2.inverse(rand_out_data2)
        rand_refine_steps2.append(rand_out_data2)
    
    if show_refinement == True:
        plt.figure()
        plt.scatter(test_data[:,0], test_data[:,1], s =10)
        #plt.scatter(out_data[:,0], out_data[:,1], s =10)
        plt.scatter(rand_out_data2[:,0], rand_out_data2[:,1], s =10)
        
    accessory_dict['refinement_steps'] = rand_refine_steps2

    
    if animate_refinement == True:
        # First set up the figure, the axis, and the plot element we want to animate
        figr3, axr3 = plt.subplots()

        axr3.set_xlim(( -1, 1))
        axr3.set_ylim((-1, 1))

        liner3, = axr3.plot([], [], lw=2, ls = '', marker = 'o', alpha = 0.2)

        def init3():
            liner3.set_data([], [])
            return (liner,)
        def animate3(i):
            d3 = rand_refine_steps2[i]
            x3,y3 = d3[:,0], d3[:,1]
            liner3.set_data(x3, y3)
            return (liner3,)
        # call the animator. blit=True means only re-draw the parts that 
        # have changed.
        animr3 = animation.FuncAnimation(figr3, animate3, init_func=init3,
                                       frames=1250, interval=20, blit=True)
        accessory_dict['animation'] = animr3
    return rand_out_data2, accessory_dict

def get_density_gradient_net(sample, D, n_samples, density_cutoff, n_bkg_samples, n_bulk_samples, density_approx = 10, sample_grid_res = 200, bandwidth=0.5, gn = 0.1):
    np.random.shuffle(D)
    D = D[::density_approx]
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(D)
    R = kde.sample(n_bulk_samples)
    
    
    xgrid = np.linspace(np.floor(D[:,0].min()),np.ceil(D[:,0].max()),sample_grid_res)
    ygrid = np.linspace(np.floor(D[:,1].min()),np.ceil(D[:,1].max()),sample_grid_res)
    X,Y = np.meshgrid(xgrid, ygrid)
    xy = np.vstack((X.ravel(), Y.ravel())).T

    Z = kde.score_samples(xy).reshape(X.shape)

    dY, dX = np.gradient(Z)
    
    dZ = np.hypot(dY,dX)*np.where(Z < density_cutoff, 0, 1)*np.where(Z> (density_cutoff+3), 0, 1)

    dZ = dZ/np.sum(dZ)

    dZ = dZ.reshape(xy.shape[0])

    draw = np.random.choice(np.arange(xy.shape[0]), n_samples,
                  p=dZ, replace = True)

    kdgrad = xy[draw] + gn*(np.random.random((n_samples, 2))-0.5*np.ones((n_samples, 2)))
    
    
    
    xmin, xmax = np.floor(np.min(sample.encoded_data[:,0])), np.ceil(np.max(sample.encoded_data[:,0]))
    ymin, ymax = np.floor(np.min(sample.encoded_data[:,1])), np.ceil(np.max(sample.encoded_data[:,1]))
    
    print(xmin, xmax,ymin,ymax)
    
    bbox = np.array(((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)))
    s_samples = np.random.random((n_bkg_samples, 2))
    
    s_samples[:,0] *= np.abs((xmax - xmin))
    s_samples[:,1] *= np.abs((ymax - ymin))
    s_samples = s_samples + np.array((xmin, ymin))
    
    return np.concatenate((kdgrad, s_samples, R), axis = 0)

def SSI_remesh(sample, R, n_add_points = 1, ssi_thresh = 0.95):
    tri = Delaunay(R)

    all_simps = tri.simplices

    line_segs= np.asarray([[np.asarray(x) for x in itertools.combinations(R[simps], 2)] for simps in all_simps])

    line_add_points= np.asarray([np.asarray([line_interp(x[0], x[1],2+n_add_points)[1:-1] for x in itertools.combinations(R[simps], 2)]) for simps in all_simps])

    f_line_segs = flatten_nav(line_segs)

    f_add_points = flatten_nav(line_add_points)

    fline_start, fline_finish = f_line_segs[:,0,:], f_line_segs[:,1,:] 

    patts_start, patts_finish = get_terr_patts(fline_start), get_terr_patts(fline_finish)

    line_ssi = np.zeros(patts_start.shape[0])

    ssi_input_data = []
    for i in range(patts_start.shape[0]):
        ssi_input_data.append((i, patts_start[i], patts_finish[i]))
        
    print('getting line ssi')

    with concurrent.futures.ProcessPoolExecutor() as exe:
        res = [exe.submit(compare_point_SSI, ssi_input) for ssi_input in ssi_input_data]
    r_batches = [f.result() for f in res]
    
    print('done getting line ssi')

    for each in r_batches:
        line_ssi[each[0]] = each[1]

    poor_line_locs = np.where(line_ssi < ssi_thresh)

    new_points = flatten_nav(f_add_points[poor_line_locs])

    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.5, c = 'grey')
    plt.triplot(R[:,0], R[:,1], all_simps, lw = 1)
    plt.scatter(new_points[:,0], new_points[:,1], s = 10, alpha = 1, c = 'black', marker = 'x')
    plt.title('Additional Mesh Points')
    
    Rp = np.concatenate((R, new_points), axis = 0)
    
    trip = Delaunay(Rp)
    
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.5, c = 'grey')
    plt.triplot(Rp[:,0], Rp[:,1], trip.simplices, lw = 1)
    plt.title('New Mesh')
    return Rp

def compare_point_SSI(required_data):
    return (required_data[0], SSI(required_data[1], required_data[2]))

from scipy.spatial import Delaunay

def latspace_from_R(R):
    return R[:,0] + 1j*R[:,1]

def get_mesh_gradients(R, rand_gradient_step, rand_n_rsteps, bs = 256):
    rand_latspace = latspace_from_R(R)

    rand_radial_kernel = rand_gradient_step* np.exp(1j*np.pi*(np.linspace(0, 360, rand_n_rsteps+1)/180))[1:]

    rand_grad_p1, rand_decdat1 = get_grad_and_decode_data(rand_latspace, rand_radial_kernel)

    return batch_calc_grad(rand_grad_p1, rand_radial_kernel, rand_decdat1, SSI_weighting, bs)

def get_mesh_transform(R, rand_linsteps):
    rand_latspace = latspace_from_R(R)
    rand_op1, rand_np1, rand_current_ps1= adjust_encoding(rand_latspace, rand_linsteps, rand_latspace, np.where(rand_latspace != None))

    rand_o_latspacer = np.asarray(rand_op1).T
    rand_n_latspacer = np.asarray(rand_np1).T

    rand_tform2 = PiecewiseAffineTransform()
    rand_tform2.estimate(rand_n_latspacer, rand_o_latspacer)
    R_moves =  (rand_np1, rand_op1)
    return rand_tform2, R_moves

def plot_R_movement(sample, R_moves):
    X,Y = (R_moves[1][0], R_moves[1][1])
    U, V = (R_moves[0][0] -  R_moves[1][0], R_moves[0][1]- R_moves[1][1])
    plt.figure(figsize = (8,8))
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.2, c = 'grey')
    plt.scatter(X, Y ,s =10, c='blue')
    plt.scatter(R_moves[0][0], R_moves[0][1],s =10, c ='orange')
    plt.quiver(X, Y, U,V)
    
def plot_enc_movement(sample, tform):
    rand_out_data2 = tform.inverse(sample.encoded_data)
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s =10)
    #plt.scatter(out_data[:,0], out_data[:,1], s =10)
    plt.scatter(rand_out_data2[:,0], rand_out_data2[:,1], s =10)
    
def repeat_tform(tform, n_refine_steps, iter_seq):
    rand_out_data2 = iter_seq[-1]
    import time
    t1 = time.time()
    for i in range(n_refine_steps):
        print(i)
        rand_out_data2 = tform.inverse(rand_out_data2)
        iter_seq.append(rand_out_data2)
    print(time.time() - t1)
    return iter_seq

def cosine_rule(a, b, c):
    return np.arccos(((a**2) +(b**2) - (c**2))/(2*a*b))

def angles(ps):
    p1, p2, p3 = ps
    a, b, c = p2-p1, p3-p2, p1-p3
    al, bl, cl = np.linalg.norm(p2-p1), np.linalg.norm(p3-p2), np.linalg.norm(p1-p3)
    return [np.rad2deg(x) for x in [cosine_rule(al, bl, cl), cosine_rule(bl, cl, al), cosine_rule(cl,al,bl)]]

def heron(ps):
    p1, p2, p3 = ps
    a, b, c = np.linalg.norm(p2-p1), np.linalg.norm(p3-p2), np.linalg.norm(p1-p3)
    s = (a+b+c)/2
    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def line_interp(p1, p2, nsteps):
    return np.concatenate([np.linspace(p1[0], p2[0], nsteps)[:,None], np.linspace(p1[1], p2[1], nsteps)[:,None]], axis = 1)

def remesh(sample, R, R_moves, density_cutoff, n_add_points = 1, n_movement_bins = 10, area_thresh = None, angle_thresh = 15, n_area_bins = 1000, n_line_interp = 10):
    #get current triangulation 
    tri = Delaunay(R)
    #view current triangulation
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.5, c = 'grey')
    plt.triplot(R[:,0], R[:,1], tri.simplices, lw = 1)
    plt.plot(R[:,0], R[:,1], 'o', markersize= 1)
    plt.title('Initial Triangulation')
    #get current mesh movements
    rand_np1, rand_op1 = R_moves
    movement = np.asarray((rand_np1[0] -  rand_op1[0], rand_np1[1]- rand_op1[1])).T
    #get total movement of the simplex vertices
    simp_move = np.sum(np.asarray([np.asarray([np.linalg.norm(movement[x]) for x in simp]) for simp in tri.simplices]), axis = 1)
    #hist these
    plt.figure()
    (n, bins, patches) =  plt.hist(simp_move, n_movement_bins)
    plt.title('Simplex Movement Histogram')
    #Truncate after first bin
    high_m_simps = tri.simplices[np.where(simp_move > bins[1])]
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.5, c = 'grey')
    plt.triplot(R[:,0], R[:,1], high_m_simps, lw = 1)
    plt.title('High Movement Simplices')
    #Get Area of remaining simplices
    simp_area = np.asarray([heron(R[simp]) for simp in high_m_simps])
    simp_angles =  np.asarray([angles(R[simp]) for simp in high_m_simps])
    plt.figure()
    (n, area_bins, patches) =  plt.hist(simp_area, n_area_bins)
    plt.title('Simplex Area Histogram')
    if area_thresh == None:
        size_inc = np.where(np.where(simp_area > area_bins[1], 1, 0) + (np.where(simp_angles[:,0] < angle_thresh, 1, 0)*np.where(simp_area > area_bins[1]/10, 1, 0))>0)
        high_a_simps = high_m_simps[size_inc]
    else: 
        size_inc = np.where(np.where(simp_area > area_thresh, 1, 0) + (np.where(simp_angles[:,0] < angle_thresh, 1, 0)*np.where(simp_area > area_thresh/10, 1, 0))>0)
        high_a_simps = high_m_simps[size_inc]
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.5, c = 'grey')
    plt.triplot(R[:,0], R[:,1], tri.simplices, lw = 1)
    plt.triplot(R[:,0], R[:,1], high_a_simps, lw = 1)
    plt.title('Area and Movement Pruned Simplices')
    #For each of the remaining mesh lines, calculate a linear interpolation of sampling points
    line_segs= np.asarray([np.asarray([line_interp(x[0], x[1],n_line_interp) for x in itertools.combinations(R[simps], 2)]) for simps in high_a_simps])
    #and a midpoint to be potentially added to the new mesh
    line_add_points= np.asarray([np.asarray([line_interp(x[0], x[1],2+n_add_points)[1:-1] for x in itertools.combinations(R[simps], 2)]) for simps in high_a_simps])
    #calculate an approximation of real data density at each of the points along the mesh line
    print(line_segs.shape)
    den_seg = np.asarray([[kde.score_samples(lss) for lss in ls] for ls in line_segs])
    #store the min and max value of this density
    den_seg_minmax = np.concatenate((den_seg.min(axis = 2)[:,:,None], den_seg.max(axis = 2)[:,:,None]), axis = 2)
    #calculate the gradient of the change in real data density along the simplex line
    den_seg_grad = np.gradient(den_seg, axis = 2)
    #Find if there is a change in sign of the density (implying a change in character of the underlying point distr)
    #old grad_changes = np.asarray([[(lines.max() * lines.min())> 0 for lines in simps] for simps in den_seg_grad])
    grad_changes = np.asarray([[ np.abs(lines).max() > 1 for lines in simps] for simps in den_seg_grad])
    #If there is a change in sign and the maximum value of the density is sufficently large 
    # ie (the line itself is near points) then add the line to be split
    interesting_lines = []
    for si in range(line_segs.shape[0]):
        for li in range(line_segs.shape[1]):
            if grad_changes[si, li] == True:
                if den_seg_minmax[si,li, 1] > density_cutoff:
                    interesting_lines.append((si, li))
    ilines = np.asarray(interesting_lines)
    i_simps =  high_a_simps[np.unique(ilines[:,0])]
    #get the midpoints of these lines
    refinement_points = flatten_nav(np.asarray([line_add_points[ninds[0], ninds[1]] for ninds in ilines]))
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.5, c = 'grey')
    plt.triplot(R[:,0], R[:,1], high_a_simps, lw = 1)
    plt.triplot(R[:,0], R[:,1], i_simps, lw = 1)
    plt.scatter(refinement_points[:,0], refinement_points[:,1], s = 10, alpha = 1, c = 'black', marker = 'x')
    plt.title('Additional Mesh Points')
    #Add these points to the original points
    Rp = np.concatenate((R, refinement_points), axis = 0)
    #View the new Triangulation
    trip = Delaunay(Rp)
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s = 10, alpha = 0.5, c = 'grey')
    plt.triplot(Rp[:,0], Rp[:,1], trip.simplices, lw = 1)
    plt.plot(Rp[:,0], Rp[:,1], 'o', markersize= 1)
    plt.title('New Triangulation')
    return (Rp, refinement_points)

def get_dense_centroids(boundaries, allowed_centroid_mask, thresh = 0.01, eps = 1.5, min_samples = 6):
    bdZ = get_grad(boundaries)

    plt.figure()
    plt.imshow(bdZ)

    plt.figure()
    plt.imshow(np.where(bdZ<thresh, 1, 0))

    plt.figure()
    plt.imshow(np.where(bdZ<thresh, 1, 0)* allowed_centroid_mask)
    
    centroid_search_region = np.where(bdZ<thresh, 1, 0)* allowed_centroid_mask

    density_stationary_points = np.asarray(np.where((centroid_search_region)==1)).T

    db = DBSCAN(eps, min_samples=min_samples )

    dspc = db.fit_predict(density_stationary_points)

    dspc.max()

    plt.figure()
    plt.scatter(density_stationary_points[:,0], density_stationary_points[:,1], c = dspc)

    dsp_centroids = np.asarray([np.mean(density_stationary_points[np.where(dspc == uind)],axis = 0) for uind in np.unique(dspc) if uind != -1])

    plt.figure()
    plt.scatter(density_stationary_points[:,0], density_stationary_points[:,1], c = dspc)
    plt.scatter(dsp_centroids[:,0], dsp_centroids[:,1], marker = 'x', c = 'red')

    xgrid = np.linspace(np.floor(D[:,0].min()),np.ceil(D[:,0].max()),sample_grid_res)
    ygrid = np.linspace(np.floor(D[:,1].min()),np.ceil(D[:,1].max()),sample_grid_res)

    centroid_approx = np.round(dsp_centroids,0).astype('int')

    centroid_approx

    espace_dsp_centroids = np.concatenate((xgrid[centroid_approx[:,1]][:,None], ygrid[centroid_approx[:,0]][:,None]), axis = 1)

    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s=5, alpha = 0.1, cmap= 'turbo')
    plt.scatter(espace_dsp_centroids[:,0], espace_dsp_centroids[:,1], marker='x')
    
    return espace_dsp_centroids, centroid_search_region


def get_sparse_centroids(boundaries, allowed_centroid_mask, thresh = 0.01, eps = 6, min_samples = 7):
    bdZ = get_grad(get_grad(get_grad(get_grad(boundaries))))

    plt.figure()
    plt.imshow(bdZ)

    plt.figure()
    plt.imshow(np.where(bdZ<thresh, 1, 0))

    plt.figure()
    plt.imshow(np.where(bdZ<thresh, 1, 0)* allowed_centroid_mask)
    
    centroid_search_region = np.where(bdZ<thresh, 1, 0)* allowed_centroid_mask

    density_stationary_points = np.asarray(np.where((centroid_search_region)==1)).T

    db = DBSCAN(eps, min_samples=min_samples )

    dspc = db.fit_predict(density_stationary_points)

    dspc.max()

    plt.figure()
    plt.scatter(density_stationary_points[:,0], density_stationary_points[:,1], c = dspc)

    dsp_centroids = np.asarray([np.mean(density_stationary_points[np.where(dspc == uind)],axis = 0) for uind in np.unique(dspc) if uind != -1])

    plt.figure()
    plt.scatter(density_stationary_points[:,0], density_stationary_points[:,1], c = dspc)
    plt.scatter(dsp_centroids[:,0], dsp_centroids[:,1], marker = 'x', c = 'red')

    xgrid = np.linspace(np.floor(D[:,0].min()),np.ceil(D[:,0].max()),sample_grid_res)
    ygrid = np.linspace(np.floor(D[:,1].min()),np.ceil(D[:,1].max()),sample_grid_res)

    centroid_approx = np.round(dsp_centroids,0).astype('int')

    centroid_approx

    espace_dsp_centroids = np.concatenate((xgrid[centroid_approx[:,1]][:,None], ygrid[centroid_approx[:,0]][:,None]), axis = 1)

    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s=5, alpha = 0.1, cmap= 'turbo')
    plt.scatter(espace_dsp_centroids[:,0], espace_dsp_centroids[:,1], marker='x')
    
    return espace_dsp_centroids, centroid_search_region

def from_centroids_refine_clusters_and_centroids(centroids, R, sample, multi = False):
    R_closest_c = find_R_closest_centroid(sample, R, centroids, multi)
    new_hull_map, probs = get_map_from_R_boundaries(R_closest_c, sample, centroids, multi = multi)
    hull_r_centroids = get_centroids(sample, new_hull_map)
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], c=flatten_nav(new_hull_map), cmap = 'turbo')
    plt.scatter(centroids[:,0], centroids[:,1], marker='x', c = 'grey')
    plt.scatter(hull_r_centroids[:,0], hull_r_centroids[:,1], marker='x', c = 'black')

    plt.figure()
    plt.imshow(new_hull_map, cmap= 'turbo')
    return hull_r_centroids, new_hull_map, probs, R_closest_c

def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)

def get_centroids(sample, map1):
    centroids = [np.mean(sample.encoded_data[np.where(flatten_nav(map1) == uind)], axis = 0) for uind in np.unique(map1)]
    return np.asarray(centroids)

def get_new_centroids(centroids, map1, ssi_thresh = 1):
    centroid_patts = get_terr_patts(centroids)

    centroid_cm = np.zeros((centroids.shape[0], centroids.shape[0]))

    for i in range((centroid_patts.shape[0])):
        for j in range((centroid_patts.shape[0])):
            if i == j:
                centroid_cm[i][j] = 100
            else:
                ssi = (1- SSI(centroid_patts[i], centroid_patts[j]))*100
                centroid_cm[i][j] = ssi


    edges = np.asarray(np.where(centroid_cm<ssi_thresh)).T
    nodes = np.unique(edges)

    g = nx.Graph()

    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    plt.figure()
    nx.draw(g, with_labels= True)

    con_comp = [x for x in nx.connected_components(g)]
    

    all_con_comps = []
    _ = [[all_con_comps.append(i) for i in e] for e in con_comp]

    uncon_comps = list(range(centroids.shape[0]))
    _ = [uncon_comps.pop(uncon_comps.index(i)) for i in all_con_comps]
    
    print(all_con_comps, uncon_comps)

    comb_map1 = np.zeros_like(map1)

    for i,uc  in enumerate(uncon_comps):
        comb_map1[np.where(map1 == uc)] = i
    for j, cc in enumerate(con_comp):
        for jc in cc:
            comb_map1[np.where(map1 == jc)] = i+j+1
            

    new_centroids = get_centroids(sample, comb_map1)
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s=5, alpha = 0.1, c = flatten_nav(comb_map1), cmap= 'turbo')
    plt.scatter(new_centroids[:,0], new_centroids[:,1], marker='x', c='red', s = 25)

    return new_centroids

def find_R_closest_centroid(sample, R, new_centroids, multi = False):
    R_patts = get_terr_patts(sample, R, multi = multi)

    ncp = get_terr_patts(sample, new_centroids, multi = multi)

    R_ssi = []
    for R_p in R_patts:
        R_ssi.append(np.argsort([SSI(R_p, incp) for incp in ncp]))

    R_ssi = np.asarray(R_ssi)

    best_centroid = R_ssi[:,-1]

    centroid_Rs = []
    for uind in np.unique(best_centroid):
        centroid_Rs.append(R[np.where(best_centroid==uind)])
    return np.asarray(centroid_Rs)

def get_map_from_R_boundaries(centroid_Rs, sample, new_centroids, multi = False):
    hull_labels = []
    for cind in range(len(centroid_Rs)):
        try:
            hull = ConvexHull(centroid_Rs[cind])
            hull_labels.append(np.where([point_in_hull(p,hull) for p in sample.encoded_data], 1, 0))
        except:
            hull_labels.append(np.zeros(sample.encoded_data.shape[0]))
    hull_labels = np.asarray(hull_labels)

    hull_labels = np.asarray(hull_labels)

    conflict_points = np.where(hull_labels.sum(axis = 0) > 1)[0]

    non_conflict = list(range(hull_labels.shape[1]))
    _ = [non_conflict.pop(non_conflict.index(i)) for i in conflict_points]

    conf = np.ones(hull_labels.shape[0])
            
    ncp = get_terr_patts(sample, new_centroids, multi = multi)
    conflict_patts = get_terr_patts(sample, sample.encoded_data[conflict_points], multi = multi)

    conflicting_point_probs = []

    for cpi, cp in enumerate(conflict_points):
        conflicting_cents = np.where(hull_labels[:,cp] ==1)[0]
        cp_patt = conflict_patts[cpi]
        conflicting_ssi = []
        for ccent in conflicting_cents:
            conflicting_ssi.append(SSI(ncp[ccent], cp_patt))
        conflicting_ssi = np.asarray(conflicting_ssi)
        probabilities = conflicting_ssi/np.sum(conflicting_ssi)
        prob_dict = {}
        for i, ccent in enumerate(conflicting_cents):
            prob_dict[ccent] = probabilities[i]
        conflicting_point_probs.append(prob_dict)

    conflicting_point_probs

    hull_prediction_labels = np.zeros(sample.encoded_data.shape[0])

    hull_labels.shape

    np.where(hull_labels[:,0]==1)[0][0]

    for nci in non_conflict:
        clust = np.where(hull_labels[:,nci]==1)[0]
        if len(clust) == 0:
            best_clust = -1
        else:
            best_clust = clust[0]
        hull_prediction_labels[nci] = best_clust

    for i, cpi in enumerate(conflict_points):
        probs = conflicting_point_probs[i]
        best_fit = list(probs.keys())[np.asarray(list(probs.values())).argmax()]
        hull_prediction_labels[cpi] = best_fit
    

    hull_labels.shape

    outliers = np.where(hull_prediction_labels ==-1)[0]

    outlier_patts = get_terr_patts(sample, sample.encoded_data[outliers], multi = multi)

    for i, p in enumerate(outlier_patts):
        hull_prediction_labels[outliers[i]] = np.argsort([SSI(p, incp) for incp in ncp])[-1]

    hull_prediction_relabelled = np.zeros_like(hull_prediction_labels)
    for i, ind in enumerate(np.unique(hull_prediction_labels)):
        hull_prediction_relabelled[np.where(hull_prediction_labels==ind)]=i 
        
    nav_shape = sample.raw_data.data.shape[:2]

    hull_pred_map = hull_prediction_relabelled.reshape(nav_shape)
    return hull_pred_map, conflicting_point_probs

def merge_centroids(sample,centroids, ssi_thresh = 1):
    centroid_patts = get_terr_patts(sample, centroids)

    centroid_cm = np.zeros((centroids.shape[0], centroids.shape[0]))

    for i in range((centroid_patts.shape[0])):
        for j in range((centroid_patts.shape[0])):
            if i == j:
                centroid_cm[i][j] = 100
            else:
                ssi = (1- SSI(centroid_patts[i], centroid_patts[j]))*100
                centroid_cm[i][j] = ssi


    edges = np.asarray(np.where(centroid_cm<ssi_thresh)).T
    nodes = np.unique(edges)

    g = nx.Graph()

    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    plt.figure()
    nx.draw(g, with_labels= True)

    con_comp = [x for x in nx.connected_components(g)]
    

    all_con_comps = []
    _ = [[all_con_comps.append(i) for i in e] for e in con_comp]

    uncon_comps = list(range(centroids.shape[0]))
    _ = [uncon_comps.pop(uncon_comps.index(i)) for i in all_con_comps]


    new_centroids = []

    for uc  in uncon_comps:
        new_centroids.append(centroids[uc])
    for j, cc in enumerate(con_comp):
        new_centroids.append(np.mean(np.asarray([centroids[jc] for jc in cc]), axis = 0))
            

    new_centroids = np.asarray(new_centroids)
    plt.figure()
    plt.scatter(sample.encoded_data[:,0], sample.encoded_data[:,1], s=5, alpha = 0.1, cmap= 'turbo')
    plt.scatter(new_centroids[:,0], new_centroids[:,1], marker='x', c='red', s = 25)

    return new_centroids

from sklearn.cluster import DBSCAN
import networkx as nx
from scipy.spatial import ConvexHull, convex_hull_plot_2d


