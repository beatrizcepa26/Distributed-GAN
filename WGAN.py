
# implementacao de https://keras.io/examples/generative/wgan_gp/#create-the-wgangp-model

from __future__ import print_function, division

import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, initializers
from keras.models import Model

import keras.backend as K

import argparse
import os

import numpy as np


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Number of GPUs: ", len(physical_devices))

# allocate only as much GPU memory as needed for the runtime allocations: it starts out allocating very little memory, 
    # and as the program gets run and more GPU memory is needed, the GPU memory region is extended for the TensorFlow process
tf.config.experimental.set_memory_growth(physical_devices[0], True)




class WGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3, gp_weight=5):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
    
    

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
    

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp



    def train_step(self, real_images):

        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the following steps as laid out in the original paper:
            # 1. Train the generator and get the generator loss
            # 2. Train the discriminator and get the discriminator loss
            # 3. Calculate the gradient penalty
            # 4. Multiply this gradient penalty with a constant weight factor
            # 5. Add the gradient penalty to the discriminator loss
            # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training the discriminator for `x` more steps 
            # (typically 5) as compared to one step of the generator. Here we will train it for 3 extra steps as compared 
            # to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)
                # type(real_logits)) -> <class 'tensorflow.python.framework.ops.SymbolicTensor'>

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))


        ##### Train the generator #####

        
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        
        return {"d_loss": d_loss, "g_loss": g_loss}
    



# saves generated images
class GANMonitor(keras.callbacks.Callback):
    
    def __init__(self, latent_dim, num_img):
        self.num_img = num_img
        self.latent_dim = latent_dim


    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        
        generated_images = self.model.generator(random_latent_vectors)
        
        generated_images *= 255
        generated_images.numpy()

        preview_dir = './wgan'

        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir, exist_ok=True)
        
        for i in range(self.num_img):
            img = keras.utils.array_to_img(generated_images[i])
            img.save(preview_dir+"/_img_%03d_%d.png" % (epoch, i))
        

# # create checkpoints
# class Checkpoint(keras.callbacks.Callback):
    
#     def __init__(self, generator, discriminator, generator_opt, discriminator_opt, dir):
#         self.generator = generator
#         self.discriminator = discriminator
#         self.generator_opt = generator_opt
#         self.discriminator_opt = discriminator_opt
#         self.dir = dir


#     def on_epoch_end(self, epoch):
        
#         checkpoint = tf.train.Checkpoint(generator_opt=self.generator_opt,
#                                         discriminator_opt=self.discriminator_opt,
#                                         generator=self.generator,
#                                         discriminator=self.discriminator,
#                                         model = self.model
#                                         )

#         ckpt_manager = tf.train.CheckpointManager(checkpoint, self.dir, max_to_keep=5)

#         # save checkpoint every 10 epochs
#         if epoch % 10 == 0:
#             ckpt_manager.save()
    

# class SaveModel(keras.callbacks.Callback):
    
#     def save_model(self):




def main():
   
    parser = argparse.ArgumentParser(description='Keras example: WGAN')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--dataset', '-i', default='',
                        help='Directory of image files.')
    parser.add_argument('--n_hidden', '-n', type=int, default=128,
                        help='Number of hidden units (z)')
    parser.add_argument('--checkpoint', '-c', action='store_true',
                        help='Restore lastest checkpoint.')

    args = parser.parse_args()

    
    # distribution strategy
    slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
    mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=slurm_resolver)

    ''' MultiWorkerMirroredStrategy implements synchronous training, where the steps of training are synced across the workers and replicas
        
        All workers train over different slices of input data in sync, and aggregating gradients at each step 
        
        tf.distribute autoshards the input dataset in multi-worker training with MultiWorkerMirroredStrategy. Autosharding a dataset over a set 
        of workers means that each worker is assigned a subset of the entire dataset (default). This is to ensure that at each step, a global 
        batch size of non-overlapping dataset elements will be processed by each worker'''
    
    print('Number of replicas:', mirrored_strategy.num_replicas_in_sync)  
    
    # create dataset from folder
    dataset = keras.utils.image_dataset_from_directory(
        args.dataset, label_mode=None, image_size=(256, 256), batch_size=args.batchsize)

    # normalize the images 
    dataset = dataset.map(lambda x: x / 255.0)

    with mirrored_strategy.scope():

        generator = keras.models.Sequential(
            [
                keras.Input(shape=(args.n_hidden)),
                
                layers.Dense(4 * 4 * 1024, 
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #l0
                layers.BatchNormalization(synchronized=True), #bn0
                layers.Reshape((4, 4, 1024)),
                
                layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same",
                                       kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc1
                layers.BatchNormalization(synchronized=True), #bn1
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same",
                                       kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc2
                layers.BatchNormalization(synchronized=True), #bn2
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same",
                                       kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc3
                layers.BatchNormalization(synchronized=True), #bn3
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same",
                                       kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc4
                layers.BatchNormalization(synchronized=True), #bn4
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding="same",
                                       kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc5
                layers.BatchNormalization(synchronized=True), #bn5
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding="same",
                                       kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc6
                layers.BatchNormalization(synchronized=True), #bn6
                layers.Dropout(0.2),
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding="same",
                                       kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #dc7
                layers.Dropout(0.2),
                layers.Activation(activations.tanh),
            ],
            name="generator",
            )
        generator.summary()


        # create the discriminator
        discriminator = keras.models.Sequential(
            [
                keras.Input(shape=(256, 256, 3)),

                layers.Conv2D(32, kernel_size=4, strides=2, padding="same",
                            kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c0_0
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(64, kernel_size=4, strides=2, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c0_1   
                layers.BatchNormalization(synchronized=True), #bn0_1
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(128, kernel_size=4, strides=2, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c1_0   
                layers.BatchNormalization(synchronized=True), #bn1_0
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(256, kernel_size=4, strides=2, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c1_1   
                layers.BatchNormalization(synchronized=True), #bn1_1
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(256, kernel_size=3, strides=1, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c2_0   
                layers.BatchNormalization(synchronized=True), #bn2_0
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(512, kernel_size=4, strides=2, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c2_1   
                layers.BatchNormalization(synchronized=True), #bn2_1
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(512, kernel_size=3, strides=1, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c3_0
                layers.BatchNormalization(synchronized=True), #bn3_0
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(1024, kernel_size=4, strides=2, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c3_1   
                layers.BatchNormalization(synchronized=True), #bn3_1
                layers.LeakyReLU(alpha=0.2),

                layers.Conv2D(1024, kernel_size=3, strides=1, padding="same",
                              kernel_initializer=initializers.RandomNormal(mean=0,stddev=0.02)), #c4_0   
                layers.BatchNormalization(synchronized=True), #bn4_0
                layers.Activation(activations.sigmoid),

                layers.Flatten(),
                layers.Dense(1) #l5
            ],
            name="discriminator",
            )
        discriminator.summary()

        
        wgan = WGAN(discriminator=discriminator, generator=generator, latent_dim=args.n_hidden, discriminator_extra_steps=3, gp_weight=5)


        # Define the loss functions for the discriminator, which should be (fake_loss - real_loss).
        # We will add the gradient penalty later to this loss function.
        def discriminator_loss(real_img, fake_img):
            real_loss = tf.reduce_mean(real_img)
            fake_loss = tf.reduce_mean(fake_img)
            return fake_loss - real_loss


        # Define the loss functions for the generator.
        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)


        # Compile the wgan model
        wgan.compile(
            d_optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5),
            g_optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.5),
            g_loss_fn=generator_loss,
            d_loss_fn=discriminator_loss,
)
       
        # create backup directory
        backup_dir = './wgan-backup'

        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir, exist_ok=True)

        backup = keras.callbacks.BackupAndRestore(backup_dir = backup_dir, 
                                                save_freq="epoch", 
                                                delete_checkpoint=False
                                                )
        
        # calculate number of batches of training
        if args.epoch % args.batchsize == 0:
            num_batches = args.epoch / args.batchsize
        else:
            num_batches = (args.epoch // args.batchsize)+1

        # save the model after the training is complete
        save = keras.callbacks.ModelCheckpoint(filepath = './wgan.keras',
                                        monitor="loss",
                                        verbose=0,
                                        save_best_only=False,
                                        save_weights_only=False,
                                        mode="auto",
                                        save_freq=num_batches,
                                        initial_value_threshold=None,
                                    )
        
        # verbose=1 gives progress bar
        wgan.fit(dataset, epochs=args.epoch, callbacks=[GANMonitor(num_img=1, latent_dim=args.n_hidden), 
                                                        backup, 
                                                        save])
        
        # restore latest checkpoint for fault tolerance
        # if args.checkpoint:
        #     wgan.fit(dataset, epochs=args.epoch, callbacks=[GANMonitor(num_img=1, latent_dim=args.n_hidden), backup])
        #     print ('Latest checkpoint restored!!')

        # wgan.fit(dataset, epochs=args.epoch, callbacks=[GANMonitor(num_img=1, latent_dim=args.n_hidden),
        #                                                 Checkpoint(generator = generator, 
        #                                                            discriminator = discriminator, 
        #                                                            generator_opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5),
        #                                                            discriminator_opt = keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.5),
        #                                                            dir = './wgan-checkpoints')]



if __name__ == '__main__':
    main()