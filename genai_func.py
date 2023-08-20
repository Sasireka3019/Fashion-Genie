# Build the metadata encoder
import tensorflow as tf
def build_metadata_encoder(num_metadata_features):
    latent_dim = 100
    metadata_input = tf.keras.layers.Input(shape=(num_metadata_features,))
    x = tf.keras.layers.Dense(128, activation='relu')(metadata_input)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    metadata_embedding = tf.keras.layers.Dense(latent_dim)(x)
    return tf.keras.Model(inputs=metadata_input, outputs=metadata_embedding)

# Build the image embedding encoder
def build_image_encoder():
    latent_dim = 100
    outfit_image = tf.keras.layers.Input(shape=(5, 64, 64, 3))

    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(outfit_image)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)

    image_embedding = tf.keras.layers.Dense(latent_dim)(x)

    return tf.keras.Model(inputs=outfit_image, outputs=image_embedding)

image_encoder = build_image_encoder()