import io
import os
import tempfile
import threading
import time
import face_recognition
import cv2
import numpy as np
import firebase_admin

from datetime import datetime
from firebase_admin import credentials, db, storage
from PIL import Image


video_capture = cv2.VideoCapture(0)

# imgAttariak = face_recognition.load_image_file('ref.jpg')
# imgAttariak_encoding = face_recognition.face_encodings(imgAttariak)[0]

# imgAli = face_recognition.load_image_file('ref1.jpg')
# imgAli_encoding = face_recognition.face_encodings(imgAli)[0]

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
def get_formated_time():
    timestamp = time.time()
    # Convert the Unix timestamp to a datetime object
    dt_object = datetime.fromtimestamp(timestamp)
    # Format the datetime object as a string in the desired format
    formatted_time = dt_object.strftime("%d-%m-%Y *** %H:%M:%S")
    
    return formatted_time

def fetch_users_from_firebase():
    ref = db.reference("/Users")
    # Reference to the Firebase Storage bucket
    bucket = storage.bucket()
    
    results = [item for item in ref.get() if item is not None]
    
    for item in results:
        known_face_names.append(item["name"])
        url = item["photo"]
        # Split the URL based on "UsersPhotos"
        parts = url.split("UsersPhotos/")
    
        # Path to the image in Firebase Storage
        image_path = f'UsersPhotos/{parts[1]}'
        # Download the image from Firebase Storage
        blob = bucket.blob(image_path)
        image_data = blob.download_as_bytes()
        
        # Save the image data to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_filename = temp_file.name
        temp_file.write(image_data)
        temp_file.close()

        imgAttariak = face_recognition.load_image_file(temp_filename)

        imgAttariak_encoding = face_recognition.face_encodings(imgAttariak)[0]
        known_face_encodings.append(imgAttariak_encoding)
        os.remove(temp_filename)
    
    

def send_user_to_firebase(photo, user):
    print(f"Sending a to firebase, user: {user['name']}")
    
    # Convert the frame to a JPEG image (you can adjust the quality if needed)
    _, img_encoded = cv2.imencode('.png', photo)
    img_bytes = img_encoded.tobytes()
    
    bucket = storage.bucket()

    
    # Save the image data to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_filename = temp_file.name
    temp_file.write(img_bytes)
    temp_file.close()
    
    # Path to the photo you want to upload
    photo_path = temp_filename

    # Destination in Cloud Storage
    destination_blob_name = f"photos/photo{time.time()}.png"

    # Upload the photo to Firebase Cloud Storage
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(photo_path)
    
    # Reference to the Realtime Database location to store photo metadata
    ref = db.reference("history")

    # Define the photo metadata
    photo_metadata = {
        "url": blob.public_url,  # The URL of the uploaded photo
        "name": user["name"],
        "time": user["time"]
        }

    # Push the metadata to the database
    new_photo_ref = ref.push(photo_metadata)
    os.remove(temp_filename)
    
def process_video():
    process_this_frame = True
    last_recorded_name = ""
    
    # Set the duration of the timer in seconds
    duration = 15
    # Start the timer
    start_time = time.time()
    while True:
        current_time = time.time()
        elapsed_time = current_time - start_time
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Only process every other frame of video to save time
        if process_this_frame:
            # Make a copy of the frame
            frame_copy = frame.copy()
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                print(elapsed_time)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if time.time() - start_time > duration or name != last_recorded_name:
                        last_recorded_name = name
                        user = {
                            "name": name,
                            "time": get_formated_time()
                        }
                        # send_user_to_firebase(frame_copy, user)
                        my_thread = threading.Thread(target=send_user_to_firebase, args=(frame_copy, user))
                        my_thread.start()
                        start_time = time.time()
                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    cred = credentials.Certificate('credentials.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://systemsecuritybebi-default-rtdb.europe-west1.firebasedatabase.app/',
        'storageBucket': 'systemsecuritybebi.appspot.com'
    })
    
    ref = db.reference("/")
    
    fetch_users_from_firebase()
    
    
    process_video()
    