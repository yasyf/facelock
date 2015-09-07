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

### Signup Flow

1. User logs in with Facebook on the iOS App
2. App sends Facebook access token to server
3. Server uses access token to determine the user's Facebook ID, and checks whether they are on the whitelist of users
4. The server then uses the Duo Auth API to determine enrollment status for that user (Facebook ID as username)
5. If the device is not enrolled, an activation code is returned, and sent back to the device
6. The iOS App uses the Duo Mobile SDK to activate and enroll the device

### Auth Flow

1. User is recognized by the images streamed to the server
2. Server uses Duo Auth API to ensure user is enrolled, then creates an authorization request
3. iOS app receives push, uses Duo Mobile SDK to get transaction info, and presents a notification to the user
4. iOS app uses Duo Mobile SDK to confirm or reject the authorization
5. If confirmed, door is unlocked
6. If rejected, the user is taken to the app, where the photo is presented (for verification and online learning)

## Hardware

A Raspberry Pi is used to communicate with the server running the above code.
