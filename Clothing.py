'''
Eric Hayter
Jan 22, 2022.

Another simple machine learning model that can tell what type of clothing a picture is.
'''

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

# storing the meaning of the labels in an array
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot'
]

# simplfying the luminosity of the pictures to be a value between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# LOOKING AT THE DATA
# showing the first 25 images of the dataset
# sets the wdith of the created window
plt.figure(figsize=(10, 10))
for i in range(25):
    # organizes the pictures into a 5x5 grid
    plt.subplot(5, 5, i + 1)
    # removing the ticks from the chart
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # makes a picture from the arrays
    # cmap will change the color of the photos to black and white instead of the default green and blue
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # puts a label on the images
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# TRAINING THE MODEL
model = tf.keras.Sequential([
    # flattening our intial 28x28 image into a 784 character long array
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    # finish with the final layer being a 10 since there are 10 possible outputs
    tf.keras.layers.Dense(10)
])

model.compile(
    # The optimizer helps optimize the model along with the loss function
    optimizer='adam',
    # The loss function helps how accurate the model is during training. This will help "steer" the model in the right direction.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

# train our model using the training input and outputs.
# Training is repeated 10 times per test to allow for maximum training without overfitting.
model.fit(train_images, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest Accuracy:', test_acc)

# once the model is trained you can use the model to predict individual pieces of data like this.
# here we use the same model as a layer but also add on the softmax which tells us the machine's confidence in each
# answer.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model(test_images)
# This will show the machine's confidence in each choiceF
print(predictions[0])


# function to print out the pictures of test data
# it will then write underneath the prediction the percent confidence
# and the true label of the image in blue if it is correct and red if it is wrong
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# prints out a bar graph beside the picture to show the confidence in each option
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()