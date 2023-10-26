import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_vae_model(hparams, dual = False, sup_data_shape = 1):

    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    def custom_loss(x,y,n_img=128):
        return tf.reduce_mean(keras.losses.binary_crossentropy(x, y)) * n_img * n_img
    if dual ==True:
        eds_shape = sup_data_shape

        n_img = 128
        latent_dim = hparams['LAT']
        beta = hparams['B']

        image_input = keras.Input(shape=(n_img, n_img,1), name = 'enc_input')
        eds_input = keras.Input(shape=(eds_shape), name = 'eds_input')
        x = layers.Conv2D(hparams['KN1'],5, strides = 2, activation='relu',padding='same', input_shape=image_input.shape, name = 'enc_conv1')(image_input)
        x = layers.Conv2D(hparams['KN2'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv2')(x)
        x = layers.Conv2D(hparams['KN3'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv3')(x)
        x = layers.Conv2D(hparams['KN4'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv4')(x)
        x = layers.Conv2D(hparams['KN5'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv5')(x)
        x = layers.Flatten()(x)
        c = layers.concatenate((x, eds_input),axis = -1)
        x = layers.Dense(hparams['D1'], activation='relu', name = 'enc_d1')(c)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d2_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d3_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d4_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d5_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d6_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d7_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d8_t')(x)
        z_mean = layers.Dense(latent_dim, name="z_mean_t")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var_t")(x)
        z_output = Sampling()([z_mean, z_log_var])
        encoder_VAE = keras.Model([image_input, eds_input], [z_mean, z_log_var, z_output])

        print(encoder_VAE.summary())

        z_input = keras.Input(shape=(latent_dim,), name = 'dec_input_t')
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d1_t')(z_input)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d2')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d3')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d4')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d5')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d6')(x)
        eds_out = layers.Dense(eds_shape, name = 'eds_out')(x)
        x = layers.Dense(hparams['D1'], activation="relu", name = 'dec_d7')(x)
        x = layers.Dense(4*4*hparams['KN5'], activation="relu", name = 'dec_d8')(x)
        x = layers.Reshape((4, 4,hparams['KN5']))(x)
        x = layers.Conv2DTranspose(hparams['KN4'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv1')(x)
        x = layers.Conv2DTranspose(hparams['KN3'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv2')(x)
        x = layers.Conv2DTranspose(hparams['KN2'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv3')(x)
        x = layers.Conv2DTranspose(hparams['KN1'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv4')(x)
        image_output = layers.Conv2DTranspose(1,5, strides = 2, activation='sigmoid',padding='same', name = 'dec_conv5')(x)
        #image_output = layers.Conv2DTranspose(16,3, strides = 2, activation='sigmoid',padding='same')
        #image_output = layers.Reshape((n_img, n_img,1))(x)
        decoder_VAE = keras.Model(z_input, [image_output,eds_out])
        print(decoder_VAE.summary())

        # VAE class
        class VAE(keras.Model):
            # constructor
            def __init__(self, encoder, decoder, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.encoder = encoder
                self.decoder = decoder

            # customise train_step() to implement the loss 
            def train_step(self, inp):
                if isinstance(inp, tuple):
                    xs = inp[0]
                    y = inp[1]
                    x = xs['enc_input']
                    eds = xs['eds_input']
                    print('dict')
                with tf.GradientTape() as tape:

                    # encoding
                    z_mean, z_log_var, z = self.encoder(xs)
                    # decoding
                    x_prime, eds_out = self.decoder(z)
                    # reconstruction error by binary crossentropy loss
                    reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime)) * n_img * n_img

                    eds_loss = tf.reduce_mean(keras.losses.mse(eds, eds_out))
                    # KL divergence
                    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                    # loss = reconstruction error + KL divergence
                    loss = reconstruction_loss + (beta* kl_loss) + eds_loss
                # apply gradient
                grads = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                # return loss for metrics log
                return {"loss": loss}

            def test_step(self, inp):
                if isinstance(inp, tuple):
                    xs = inp[0]
                    y = inp[1]
                    x = xs['enc_input']
                    eds = xs['eds_input']
                    print('dict')

                # encoding
                z_mean, z_log_var, z = self.encoder(xs)
                print('encoded')
                # decoding
                x_prime, eds_out = self.decoder(z)
                print('decoded')
                # reconstruction error by binary crossentropy loss
                reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_prime)) * n_img * n_img
                print('recon loss')
                print(eds, eds_out)
                eds_loss = tf.reduce_mean(keras.losses.mse(eds, eds_out))
                print('eds loss')
                # KL divergence
                kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                print('kl loss')
                # loss = reconstruction error + KL divergence
                loss = reconstruction_loss + (beta* kl_loss) + eds_loss

                return {"loss": loss}


            def call(self, xs):
                # encoding
                z_mean, z_log_var, z = self.encoder(xs)
                # decoding
                x_prime = self.decoder(z)
                return x_prime

        # build the VAE
        vae_model = VAE(encoder_VAE, decoder_VAE)

        # compile the VAE
        vae_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hparams['LR']),loss=custom_loss)
        vae_model.build([(1,128,128,1),(1,4000)])

        return vae_model
    else:
        n_img = 128
        latent_dim = hparams['LAT']
        beta = hparams['B']

        image_input = keras.Input(shape=(n_img, n_img,1), name = 'enc_input')
        x = layers.Conv2D(hparams['KN1'],5, strides = 2, activation='relu',padding='same', input_shape=image_input.shape, name = 'enc_conv1')(image_input)
        x = layers.Conv2D(hparams['KN2'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv2')(x)
        x = layers.Conv2D(hparams['KN3'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv3')(x)
        x = layers.Conv2D(hparams['KN4'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv4')(x)
        x = layers.Conv2D(hparams['KN5'],5, strides = 2, activation='relu',padding='same', name = 'enc_conv5')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(hparams['D1'], activation='relu', name = 'enc_d1')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d2_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d3_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d4_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d5_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d6_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d7_t')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'enc_d8_t')(x)
        z_mean = layers.Dense(latent_dim, name="z_mean_t")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var_t")(x)
        z_output = Sampling()([z_mean, z_log_var])
        encoder_VAE = keras.Model(image_input, [z_mean, z_log_var, z_output])

        z_input = keras.Input(shape=(latent_dim,), name = 'dec_input_t')
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d1_t')(z_input)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d2')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d3')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d4')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d5')(x)
        x = layers.Dense(hparams['D2'], activation="relu", name = 'dec_d6')(x)
        x = layers.Dense(hparams['D1'], activation="relu", name = 'dec_d7')(x)
        x = layers.Dense(4*4*hparams['KN5'], activation="relu", name = 'dec_d8')(x)
        x = layers.Reshape((4, 4,hparams['KN5']))(x)
        x = layers.Conv2DTranspose(hparams['KN4'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv1')(x)
        x = layers.Conv2DTranspose(hparams['KN3'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv2')(x)
        x = layers.Conv2DTranspose(hparams['KN2'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv3')(x)
        x = layers.Conv2DTranspose(hparams['KN1'],5, strides = 2, activation='relu',padding='same', name = 'dec_conv4')(x)
        image_output = layers.Conv2DTranspose(1,5, strides = 2, activation='sigmoid',padding='same', name = 'dec_conv5')(x)
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
                    loss = reconstruction_loss + beta* kl_loss
                # apply gradient
                grads = tape.gradient(loss, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                # return loss for metrics log
                return {'{"loss": loss}'}


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
        vae_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hparams['LR']),loss=custom_loss)
        vae_model.build((1,128,128,1))

        return vae_model
