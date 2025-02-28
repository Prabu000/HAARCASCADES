import cv2
import os
import time


eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


video_capture = cv2.VideoCapture(0)


output_dirs = {
    "eyes": "captured_eyes",
    "faces": "captured_faces",
    "bodies": "captured_bodies",
    "smiles": "smile_photos"
}


for dir_name in output_dirs.values():
    os.makedirs(dir_name, exist_ok=True)


def detect_eyes(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)  
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  
    eyes = eye_classifier.detectMultiScale(gray_image, 1.1, 10, minSize=(20, 20))
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Eye', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return eyes


def detect_faces(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return faces


def detect_bodies(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image) 
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0) 
    bodies = body_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
        cv2.putText(frame, 'Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return bodies


def detect_smiles(frame):
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image) 
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  
    faces = face_classifier.detectMultiScale(gray_image)
    smile_detected = False
    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        smiles = smile_classifier.detectMultiScale(roi_gray, 1.7, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx + x, sy + y), (sx + sw + x, sy + sh + y), (255, 0, 255), 4) 
            cv2.putText(roi_color, 'Smile', (sx + x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
            smile_detected = True
    return smile_detected


last_saved_time_eyes = time.time()
last_saved_time_faces = time.time()
last_saved_time_bodies = time.time()
last_saved_time_smiles = time.time()
capture_interval = 2 

while True:
    result, video_frame = video_capture.read()
    if not result:
        break


    display_width = 800 
    display_height = 600 
    video_frame = cv2.resize(video_frame, (display_width, display_height))


    eyes = detect_eyes(video_frame)
    faces = detect_faces(video_frame)
    bodies = detect_bodies(video_frame)
    smile_detected = detect_smiles(video_frame)

    current_time = time.time()


    if len(eyes) > 0 and (current_time - last_saved_time_eyes) >= capture_interval:
        image_path_eyes = os.path.join(output_dirs["eyes"], f"eye_{int(current_time)}.jpg")
        cv2.imwrite(image_path_eyes, video_frame)
        last_saved_time_eyes = current_time

    if len(faces) > 0 and (current_time - last_saved_time_faces) >= capture_interval:
        image_path_faces = os.path.join(output_dirs["faces"], f"face_{int(current_time)}.jpg")
        cv2.imwrite(image_path_faces, video_frame)
        last_saved_time_faces = current_time

    if len(bodies) > 0 and (current_time - last_saved_time_bodies) >= capture_interval:
        image_path_bodies = os.path.join(output_dirs["bodies"], f"body_{int(current_time)}.jpg")
        cv2.imwrite(image_path_bodies, video_frame)
        last_saved_time_bodies = current_time

    if smile_detected and (current_time - last_saved_time_smiles) >= capture_interval:
        image_path_smiles = os.path.join(output_dirs["smiles"], f"smile_{int(current_time)}.jpg")
        cv2.imwrite(image_path_smiles, video_frame)
        last_saved_time_smiles = current_time


    cv2.imshow("Detection Project", video_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
