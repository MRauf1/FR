from argparse import ArgumentParser
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn import neighbors
import pickle

# Command line arguments
parser = ArgumentParser(description="Train the KNN classifier")
parser.add_argument("-d", "--dir", required=True, help="Training directory with class subfolders.")
parser.add_argument("-s", "--save_path", required=True, help="Path to where the model should be saved.")
parser.add_argument("-n", "--neighbors", type=int, nargs="?", const=3, help="Number of neighbors for the KNN classifier. Default is 3.")

# Get the initial variables 
args = vars(parser.parse_args())

train_dir = args["dir"]
save_path = args["save_path"]
num_neighbors = args["neighbors"]

X = []
Y = []

# Navigate through each person's training folder
for name in os.listdir(train_dir):

    # Detect the face in the images
    for image_path in image_files_in_folder(os.path.join(train_dir, name)):
        
        image = face_recognition.load_image_file(image_path)
        bounding_boxes = face_recognition.face_locations(image)

        # Ignore images that do not have only 1 face
        if len(bounding_boxes) == 1:

            X.append(face_recognition.face_encodings(image, known_face_locations=bounding_boxes)[0])
            Y.append(name)
                
        else:

            print(str(image_path) + " does not have only 1 face. Image skipped")

# Train the KNN Classifier
knn = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, algorithm="ball_tree", weights="distance")
knn.fit(X, Y)

# Save the KNN Classifier model
with open(save_path, 'wb') as f:

    pickle.dump(knn, f)
