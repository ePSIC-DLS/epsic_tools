import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import DBSCAN
from skimage.filters import gaussian

class SegModel():
    def __init__(self,
                 model_weights = '/home/dto55534/.local/lib/python3.7/site-packages/stemseg/extras/weights/centroid_pred.hdf5'):

        # Load the pre-trained ResNet model without the top layer
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3))

        # Define the U-Net architecture
        inputs = Input(shape=(128, 128, 1))

        # Add a layer to convert the single channel input to 3 channels so that we can use the ResNet model
        x = Conv2D(3, (1, 1))(inputs) # This layer converts the input image to 3 channels
        x = base_model(x)

        # Define the decoder part of the U-Net
        x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        x = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization()(x)

        # Up-sample to the desired output size
        outputs = Conv2DTranspose(1, (4, 4), strides=(4, 4), padding='same')(x)

        # Create the model
        model = Model(inputs, outputs)

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        # Print a summary of the model
        model.load_weights(model_weights)
        
        self.model = model

    def img_to_original_coords(self, pixel_loc, real_data, img_size = (256,256)):
        lims = ((np.min(real_data[:,0]), np.max(real_data[:,0])),(np.min(real_data[:,1]), np.max(real_data[:,1]))) 
        cx, cy = pixel_loc[0]/img_size[0], pixel_loc[1]/img_size[1]
        ocx = (cx*(lims[0][1]-lims[0][0])) + lims[0][0]
        ocy = (cy*(lims[1][1]-lims[1][0])) + lims[1][0]
        return np.array((ocx, ocy))

    def predict_centroids_dbscan(self, real_img, real_data, vis = False, eps = 6, min_samples = 5, thresh = 0.5):
        real_pred = self.model.predict(real_img[None,:,:,None])[0,:,:,0]
        
        pred_coords = np.asarray(np.where(real_pred> thresh)).T

        cent_cs = DBSCAN(eps =eps, min_samples = min_samples).fit_predict(pred_coords)
        
        pred_centroids = []
        for cent_c in np.unique(cent_cs):
            pred_cent = np.mean(pred_coords[np.where(cent_cs == cent_c)], axis = 0)
            pred_centroids.append(self.img_to_original_coords(pred_cent, real_data))
            #pred_centroids.append(pred_cent)
        pred_centroids = np.asarray(pred_centroids)
        
        if vis == True:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(real_img)
            ax[1].imshow(real_pred, vmin = 0, vmax = 0.5)
            
            plt.figure()
            plt.scatter(real_data[:,0], real_data[:,1], s= 1)
            plt.scatter(pred_centroids[:,0], pred_centroids[:,1])
            
        return pred_centroids
    
    
    def predict_centroids_watershed(self, real_img, real_data, vis = False, gk = 2, thresh = 0.5):
        real_pred = self.model.predict(real_img[None,:,:,None])[0,:,:,0]
        
        image = np.where(gaussian(real_pred, gk)> thresh,1,0)
        
        from scipy import ndimage as ndi

        from skimage.segmentation import watershed
        from skimage.feature import peak_local_max


        # Now we want to separate the two objects in image
        # Generate the markers as local maxima of the distance to the background
        distance = ndi.distance_transform_edt(image)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        labels = watershed(-distance, markers, mask=image)
        
        print(labels.shape)
        
        
        pred_centroids = []
        for u_ind in np.unique(labels):
            pred_cent = np.mean(np.asarray(np.where(labels == u_ind)).T, axis = 0)
            pred_centroids.append(self.img_to_original_coords(pred_cent, real_data))
        pred_centroids = np.asarray(pred_centroids)[1:]
        
        if vis == True:
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(real_img)
            ax[1].imshow(real_pred, vmin = 0, vmax = 0.5)
            ax[2].imshow(labels)
            
            plt.figure()
            plt.scatter(real_data[:,0], real_data[:,1], s= 1)
            plt.scatter(pred_centroids[:,0], pred_centroids[:,1])
            
        return pred_centroids

    def points_to_img_repr(self, clust_points, show = True, grid_image_shape = (128, 128), threshold =1):
        
        from stemutils.visualise import get_minmax_grid_array
        from sklearn.neighbors import NearestNeighbors


        clust_size = clust_points.shape[0]

        nnsearch = NearestNeighbors(n_neighbors = clust_size//5)
        nnsearch.fit(clust_points)


        cent_clust_points = clust_points - np.mean(clust_points, axis = 0)[None, :]

        norm_clust_points = cent_clust_points/np.std(cent_clust_points, axis = 0)[None, :]

        xmin, ymin = np.min(norm_clust_points, axis = 0)
        xmax, ymax = np.max(norm_clust_points, axis = 0)

        grid_image = np.zeros(grid_image_shape)

        grid_coords = get_minmax_grid_array(((xmin, xmax, grid_image.shape[0]),
                                             (ymin, ymax,grid_image.shape[1])), True)

        print('on nearest pixel')

        nearest_pix = np.asarray([np.argmin(np.linalg.norm(grid_coords - p, axis = 1)) 
                               for p in norm_clust_points])

        print('started image processing')

        grid_image = np.zeros(grid_image_shape)

        for pix_ind in nearest_pix:
            grid_image[np.unravel_index(pix_ind, grid_image.shape)] += 1
            
        norm_img = grid_image/grid_image.max()
        
        out_img = np.where(norm_img > threshold, threshold, norm_img)/threshold

        return out_img

