import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout, LeakyReLU, BatchNormalization, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.datasets import mnist
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt


def load_dataset(csv_path, images_folder, target_size=(28, 28), grayscale=True):
    df = pd.read_csv(csv_path)
    df = df[df['6_way_label'] == 1] # here just onle fake image 
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = row['id']
        label = row['6_way_label']
        image_path = f"{images_folder}/{img_path}.jpg"
        if os.path.exists(image_path): 
            img = cv2.imread(image_path)
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else: 
            print(f"Image not found: {image_path}")
            continue  # Skip this iteration and move on to the next row

        img = cv2.resize(img, target_size)
        images.append(img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

csv_path = "../Data/multimodal_train.csv"
images_folder = "../Data/images"

# Load and preprocess the dataset
x_train, y_train = load_dataset(csv_path, images_folder)
x_train = x_train / 127.5 - 1.0
x_train = np.expand_dims(x_train, axis=-1)

# Define the generator model
def create_generator(latent_dim):
    input_noise = Input(shape=(latent_dim,))
    input_img = Input(shape=img_shape)

    x = Concatenate()([input_noise, Flatten()(input_img)])
    x = Dense(128 * 7 * 7)(x)
    x = LeakyReLU(0.2)(x)
    x = Reshape((7, 7, 128))(x)
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    output = Conv2D(1, kernel_size=3, padding='same', activation='tanh')(x)

    model = Model([input_noise, input_img], output)
    return model

# Define the discriminator model
def create_discriminator(img_shape):
    input_real = Input(shape=img_shape)
    input_cond = Input(shape=img_shape)

    x = Concatenate()([input_real, input_cond])
    x = Conv2D(32, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model([input_real, input_cond], output)
    return model


latent_dim = 100
img_shape = x_train[0].shape

# Create and compile the discriminator
discriminator = create_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Create the generator
generator = create_generator(latent_dim)

# Create the combined model (generator -> discriminator)
input_noise = Input(shape=(latent_dim,))
input_img = Input(shape=img_shape)
generated_img = generator([input_noise, input_img])
discriminator.trainable = False
validity = discriminator([generated_img, input_img])

combined = Model([input_noise, input_img], validity)
combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

losses = []

epochs = 400
batch_size = 32

for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    input_imgs = x_train[np.random.randint(0, x_train.shape[0], batch_size)] # Input images for conditioning
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict([noise, input_imgs])
    
    real_y = np.ones((batch_size, 1))
    fake_y = np.zeros((batch_size, 1))
    
    d_loss_real = discriminator.train_on_batch([real_imgs, input_imgs], real_y)
    d_loss_fake = discriminator.train_on_batch([gen_imgs, input_imgs], fake_y)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.ones((batch_size, 1))
    
    g_loss = combined.train_on_batch([noise, input_imgs], valid_y)
    
    losses.append((d_loss[0], g_loss))
    # Print progress
    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, D loss: {d_loss[0]}, G loss: {g_loss}")

# After training, plot the losses
losses = np.array(losses)

plt.figure(figsize=(10, 5))
plt.plot(losses.T[0], label='Discriminator loss')
plt.plot(losses.T[1], label='Generator loss')
plt.title('Training losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
generator.save('gan_generator_model.h5')

