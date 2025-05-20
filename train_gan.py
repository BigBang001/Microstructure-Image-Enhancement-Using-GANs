import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

def add_noise(img):
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.1)
    noisy_img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return noisy_img

def preprocess(hr_img):
    hr_img = tf.image.convert_image_dtype(hr_img, tf.float32)
    lr_img = tf.image.resize(hr_img, [16,16], method='bicubic')
    lr_img = tf.image.resize(lr_img, [32,32], method='bicubic')
    lr_noisy_img = add_noise(lr_img)
    return lr_noisy_img, hr_img

def load_cifar10(batch_size=16):
    (x_train, _), _ = tf.keras.datasets.cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.map(lambda x: preprocess(x))
    dataset = dataset.batch(batch_size).shuffle(1000).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = load_cifar10()

def residual_block(x, filters=64):
    res = layers.Conv2D(filters, 3, padding='same')(x)
    res = layers.BatchNormalization()(res)
    res = layers.Activation('relu')(res)
    res = layers.Conv2D(filters, 3, padding='same')(res)
    res = layers.BatchNormalization()(res)
    return layers.Add()([x, res])

def build_generator():
    inputs = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 9, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    res = x
    for _ in range(5):
        res = residual_block(res, 64)
    x = layers.Conv2D(64, 3, padding='same')(res)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, res])
    x = layers.Conv2D(3, 9, padding='same')(x)
    outputs = layers.Activation('sigmoid')(x)
    return Model(inputs, outputs, name='Generator')

generator = build_generator()

def build_discriminator():
    inputs = layers.Input(shape=(32,32,3))
    x = layers.Conv2D(64, 3, strides=1, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='Discriminator')

discriminator = build_discriminator()

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
l1_loss = tf.keras.losses.MeanAbsoluteError()

gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(lr_noisy, hr):
    batch_size = tf.shape(lr_noisy)[0]
    valid = tf.ones((batch_size, 1))
    fake = tf.zeros((batch_size, 1))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_hr = generator(lr_noisy, training=True)
        pred_real = discriminator(hr, training=True)
        pred_fake = discriminator(gen_hr, training=True)
        loss_GAN = bce(valid, pred_fake)
        loss_pixel = l1_loss(hr, gen_hr)
        loss_G = loss_GAN + 100 * loss_pixel
        loss_real = bce(valid, pred_real)
        loss_fake = bce(fake, pred_fake)
        loss_D = (loss_real + loss_fake) * 0.5
    grads_G = gen_tape.gradient(loss_G, generator.trainable_variables)
    grads_D = disc_tape.gradient(loss_D, discriminator.trainable_variables)
    gen_optimizer.apply_gradients(zip(grads_G, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(grads_D, discriminator.trainable_variables))
    return loss_G, loss_D

epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for step, (lr_noisy_batch, hr_batch) in enumerate(train_dataset):
        loss_G, loss_D = train_step(lr_noisy_batch, hr_batch)
        if step % 100 == 0:
            print(f"Step {step}: Generator Loss: {loss_G:.4f}, Discriminator Loss: {loss_D:.4f}")

# Save the generator model after training for demo usage
generator.save("generator_model")

def show_images(lr_noisy, gen_hr, hr):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,5))
    for i in range(min(3, lr_noisy.shape[0])):
        plt.subplot(3,3,i*3+1)
        plt.imshow(lr_noisy[i].numpy())
        plt.title("Input: Noisy Low-res")
        plt.axis('off')
        plt.subplot(3,3,i*3+2)
        plt.imshow(gen_hr[i].numpy())
        plt.title("Generated High-res")
        plt.axis('off')
        plt.subplot(3,3,i*3+3)
        plt.imshow(hr[i].numpy())
        plt.title("Ground Truth")
        plt.axis('off')
    plt.show()

for lr_noisy_batch, hr_batch in train_dataset.take(1):
    gen_hr_batch = generator(lr_noisy_batch, training=False)
    show_images(lr_noisy_batch, gen_hr_batch, hr_batch)
