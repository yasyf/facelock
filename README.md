# Facelock

An all-in-one solution for unlocking a door with two-factor facial recognition of Facebook users.

## Face Recognition

All image manipulation is done with OpenCV 2. A list of Facebook users is maintained, and a model is trained to recognize the face of each user.

Authentication is granted via Facebook's Device Authorization API, and a configurable number of photos are fetched.

The photos are first cropped to roughly the area defined by the tag on Facebook (exact dimensions are not provided in the API, only the center of the tag), so as to eliminate other people. A couple Haar classifier trained on frontal faces are then used to identify the face of the user, and the image is again cropped. The eyes of this face are then identified, and used to rotate and further crop the image so that only the aligned face remains. Finally, the image is normalized by equalizing its histogram, removing any noise from lighting and contrast differences.

Online learning is supported by tracking the correctness of predictions, and re-training the model periodically.

## iOS App

An iOS app is used for two-factor authentication. If the phone belonging to the recognized face is within a configurable distance from the door, the door is automatically unlocked. Otherwise, a push notification is sent to verify the user's identity.

The app is also used to collect feedback in order to continuously update the facial recognition model.

## Hardware

A Raspberry Pi is used to communicate with the server running the above code.
