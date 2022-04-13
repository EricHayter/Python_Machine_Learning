import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

cifar = tf.keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = cifar.load_data()

CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

model = tf.keras.models.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255,
                                                         input_shape=(32, 32,
                                                                      3)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.summary()

history = model.fit(train_images, train_labels, epochs=1)
predictions = model(test_images)

print(predictions[0])

plt.figure(figsize=(4, 5))
for x in range(20):
    plt.subplot(4, 5, x + 1)
    plt.imshow(test_images[x])
    plt.xlabel(
        f'prediction: {CLASSES[predictions[x]]}\nactual: {CLASSES[test_labels[x]]}'
    )
plt.show()

model.evaluate(x=test_images, y=test_labels)
