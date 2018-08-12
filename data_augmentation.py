from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from argparse import ArgumentParser
from os.path import dirname

# Command line arguments
parser = ArgumentParser(description="Data augment a specific image 50 times")
parser.add_argument("-p", "--path", required=True, help="Path to the image to be augmented.")

# Get the initial variables 
path = vars(parser.parse_args())["path"]
directory = dirname(path)

# Parameters/Options for data augmentation
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Retrieve the image
image = load_img(path)
X = img_to_array(image)
X = X.reshape((1,) + X.shape)

# Generate 50 new images and save them to the assigned directory
i = 0
for batch in datagen.flow(X, batch_size=1,
                          save_to_dir=directory, save_prefix='new', save_format='jpeg'):
    i += 1
    if i > 50:
        break
