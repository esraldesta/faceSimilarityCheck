import dlib
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
import cv2
import face_recognition
avg_threshold = 0.5

def check_face(request):
    if request.method == 'POST':
        image1 = request.FILES.get('image1')
        image2 = request.FILES.get('image2')
        if image1 and image2:
            result_dlib = check_dlib(image1,image2)
            result_cv2= check_cv2(image1,image2)
            result_face =check_face_recognition(image1,image2)
            theSum= 0
            count = 0
            for i in [result_dlib,result_cv2,result_face]:
                try:
                    theSum += float(i)
                    count+=1
                    print("hi",float(i))
                    print("hi",count)
                except Exception as e:
                    pass
            print("sum",theSum)
            print("count",count)
            return JsonResponse({"result":float(theSum)/float(count)})

    # Render the form to a template
    return JsonResponse({"message": "no get"})


def check_dlib(image1,image2):
    # Load the images and get the face encodings
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    image1_array = np.frombuffer(image1.file.read(), np.uint8)
    image1_array = cv2.imdecode(image1_array, cv2.IMREAD_COLOR)
    image1_gray = cv2.cvtColor(image1_array, cv2.COLOR_BGR2GRAY)
    image1_faces = detector(image1_gray)
    image1_landmarks = [predictor(image1_gray, face) for face in image1_faces]
    image1_encodings = [np.array(face_recognizer.compute_face_descriptor(image1_array, landmarks)) for landmarks in image1_landmarks]
    image2_array = np.frombuffer(image2.file.read(), np.uint8)
    image2_array = cv2.imdecode(image2_array, cv2.IMREAD_COLOR)
    image2_gray = cv2.cvtColor(image2_array, cv2.COLOR_BGR2GRAY)
    image2_faces = detector(image2_gray)
    image2_landmarks = [predictor(image2_gray, face) for face in image2_faces]
    image2_encodings = [np.array(face_recognizer.compute_face_descriptor(image2_array, landmarks)) for landmarks in image2_landmarks]
    # Compare the face encodings and calculate the similarity score
    similarity_scores = [np.linalg.norm(image1_encoding - image2_encoding) for image1_encoding in image1_encodings for image2_encoding in image2_encodings]
    similarity_score = np.mean(similarity_scores)
    # Render the result to a template
    # return render(request, 'result.html', {'similarity_score': similarity_score})
    return similarity_score

def check_cv2(image1,image2):
    try:
        # Load the images and detect faces using OpenCV
        print("here1")
        # image1_np = np.fromfile(image1, dtype=np.uint8)
        print("here2")
        # image1_cv = cv2.imdecode(image1_np, cv2.IMREAD_COLOR)
        image1_cv = cv2.imdecode(np.frombuffer(image1.read(), np.uint8), cv2.IMREAD_COLOR)
        image1_gray = cv2.cvtColor(image1_cv, cv2.COLOR_BGR2GRAY)
        image1_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(image1_gray)
        image1_encodings = []
        for (x, y, w, h) in image1_faces:
            face = image1_cv[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160)) # Resize the face image to a common size
            face = np.array(face) / 255.0 # Normalize the pixel values
            image1_encodings.append(face)
        # image2_np = np.fromfile(image2, dtype=np.uint8)
        # image2_cv = cv2.imdecode(image2_np, cv2.IMREAD_COLOR)
        image2_cv = cv2.imdecode(np.frombuffer(image2.read(), np.uint8), cv2.IMREAD_COLOR)
        image2_gray = cv2.cvtColor(image2_cv, cv2.COLOR_BGR2GRAY)
        image2_faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(image2_gray)
        image2_encodings = []
        for (x, y, w, h) in image2_faces:
            face = image2_cv[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            face = np.array(face) / 255.0
            image2_encodings.append(face)
        # Compare the face encodings and calculate the similarity score
        similarity_scores = [np.linalg.norm(image1_encoding - image2_encoding) for image1_encoding in image1_encodings for image2_encoding in image2_encodings]
        similarity_score = np.mean(similarity_scores)
        # Render the result to a template
        return similarity_score
    except Exception as e:
        print(e)


def check_face_recognition(image1,image2):
    image1_encoding = face_recognition.face_encodings(face_recognition.load_image_file(image1))[0]
    image2_encoding = face_recognition.face_encodings(face_recognition.load_image_file(image2))[0]
    # Compare the face encodings and calculate the similarity score
    results = face_recognition.compare_faces([image1_encoding], image2_encoding)
    similarity_score = face_recognition.face_distance([image1_encoding], image2_encoding)
    # Render the result to a template
    # return render(request, 'result.html', {'similarity_score': similarity_score[0]})
    return similarity_score[0]