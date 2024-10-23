import threading
import time
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

shared_data = { "x" : -1, "y" : -1}

# Cria uma condição que combina o lock com uma maneira de notificar threads
condition = threading.Condition()

def write_data():
    global shared_data
    # Substituir este bloco com o código do MediaPipe
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            img_h, img_w = image.shape[:2]
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                    
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    
                    with condition:
                        shared_data["x"], shared_data["y"] = center_left[0], center_left[1]
                        condition.notify_all()

            cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

def read_data():
    while True:
        with condition:
            condition.wait()
            # A thread só é despertada após a notificação
            print(f"Dados atualizados - Posição do olho: x={shared_data['x']}, y={shared_data['y']}")

# Criação das threads
write_thread = threading.Thread(target=write_data)
read_thread = threading.Thread(target=read_data)

# Inicia as threads
write_thread.start()
read_thread.start()

# Aguarda as threads finalizarem (nesse caso, elas rodam indefinidamente)
write_thread.join()
read_thread.join()
