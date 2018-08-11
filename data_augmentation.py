from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
image = load_img('train_images/arslan/index.jpg')
X = img_to_array(image)
X = X.reshape((1,) + X.shape)

# Generate 50 new images and save them to the assigned directory
i = 0
for batch in datagen.flow(X, batch_size=1,
                          save_to_dir='train_images/arslan', save_prefix='new', save_format='jpeg'):
    i += 1
    if i > 50:
        break