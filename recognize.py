from argparse import ArgumentParser
import face_recognition
from sklearn import neighbors
import pickle
import cv2
import time

# Command line arguments
parser = ArgumentParser(description="Recognize Faces")

parser.add_argument("-m", "--model_path", required=True, help="Path to where the model is located.")
parser.add_argument("-n", "--neighbors", type=int, nargs="?", const=3, help="Number of neighbors for the KNN classifier. Default is 3.")
parser.add_argument("-d", "--distance", type=float, nargs="?", const=0.6, help="Distance threshold for the matching via the KNN classifier. Default is 0.6.")

# Get the initial variables 
args = vars(parser.parse_args())

model_path = args["model_path"]
num_neighbors = args["neighbors"]
distance_threshold = args["distance"]

# Load the KNN classifier
with open(model_path, 'rb') as f:
    
    knn = pickle.load(f)

print("Everything loaded")

# ***For accessing the IP camera
#camera = cv2.VideoCapture("insert stream url")

# Start the webcam
video = VideoStream().start()
print("Video started")

while(True):
    
    print("Started new frame")

    # ***For accessing the IP camera
    #ret,img = cam.read()

    # Read each frame
    frame = video.read()

    # Detect the faces
    bounding_boxes = face_recognition.face_locations(frame)

    # Skip to the next frame if no faces are detected
    if len(bounding_boxes) == 0:

        print("No faces")
        continue
    
    # Get the face encodings for the faces
    encodings = face_recognition.face_encodings(frame, known_face_locations=bounding_boxes)
    
    print("Finished detecting and encoding the faces")

    # Find the matches with the KNN classifier
    closest_distances = knn.kneighbors(encodings, n_neighbors=num_neighbors)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(bounding_boxes))]

    names = knn.predict(encodings)
    
    # Print the name and the bounding box
    print([(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(names, bounding_boxes, are_matches)])
    
    # Show the name with the bounding box
    for (top, right, bottom, left), name in zip(bounding_boxes, names):

        # Outline the face with the box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Write the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the final image
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
