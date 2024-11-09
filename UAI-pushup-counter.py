import cv2
import mediapipe as mp
import os
import warnings

warnings.filterwarnings("ignore")

mp_drawing=mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
stage = None
create = None

outputfile = "processed_live_video_UAI.avi"

# Generamos una lista con los landmarks(puntos del cuerpo) con un id y coordenadas
def findPosition(image, draw=True):
    lmList = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = image.shape
            cx,cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id,cx,cy])
            #cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return lmList

# Capturamos el video con cv2

# Para capturar desde la webcam
cap = cv2.VideoCapture(1)

# Pasar path absoluto con video para usar uno ya grabado
# cap = cv2.VideoCapture(r"./pushup_video.mp4")


# Aca se crea una instancia del modelo de pose de MediaPipe con una confianza mínima para la detección y el seguimiento establecida en 0.7.
# Esto controla qué tan seguro debe estar el modelo para detectar y seguir los landmarks del cuerpo.
with mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        #Se lee la imagen de la cámara (cap) y se redimensiona a 640x480 píxeles. Si la lectura no tiene éxito, se continúa el bucle.
        success, image = cap.read()
        image = cv2.resize(image, (640, 480))
        if not success:
            print("Ignorando el frame vacio de la camara.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        #La imagen se voltea horizontalmente (para dar una vista de tipo "selfie").
        #Se convierte de BGR (formato usado por OpenCV) a RGB (requerido por MediaPipe).
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # El modelo procesa la imagen para detectar los puntos clave (landmarks) del cuerpo.
        results = pose.process(image)

        #Dibujo de los landmarks y detección de flexiones
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lmList = findPosition(image, draw=True)
        if len(lmList) != 0:
            # Dibuja círculos en los landmarks del hombro derecho (lmList[12]) e izquierdo (lmList[11])
            cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):
                cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
                stage = "Abajo"
            if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "Abajo":
                stage = "Arriba"
                counter += 1
                counter2 = str(int(counter))
                print(counter)
                # os.system("echo '" + counter2 + "' | festival --tts")
        text = "{}:{}".format("UAI - Contador de Push Ups", counter)
        cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        exit_text = "Presionar 'q' para salir. "

        # Obtener el tamaño de la imagen
        height, width, _ = image.shape

        # Calcular la posición para que el texto aparezca en la esquina inferior derecha
        text_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = width - text_size[0] - 10  # Restamos 10 para dejar un margen
        text_y = height - 10  # 10 píxeles arriba del borde inferior
        cv2.putText(image, exit_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MediaPipe Pose', image)

        if create is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            create = cv2.VideoWriter(outputfile, fourcc, 30, (image.shape[1], image.shape[0]), True)
        create.write(image)
        key = cv2.waitKey(1) & 0xFF

        # Salir del programa cuando se presione la tecla "q"
        if key == ord("q"):
            break

# Eliminamos los objetos creados por cv2
cv2.destroyAllWindows()
nahux@dharma:~ $ cat pushupcounter.py
import cv2
import mediapipe as mp
import os
import warnings

warnings.filterwarnings("ignore")

mp_drawing=mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
counter = 0
stage = None
create = None

outputfile = "processed_live_video_UAI.avi"

# Generamos una lista con los landmarks(puntos del cuerpo) con un id y coordenadas
def findPosition(image, draw=True):
    lmList = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = image.shape
            cx,cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id,cx,cy])
            #cv2.circle(image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    return lmList

# Capturamos el video con cv2

# Para capturar desde la webcam
cap = cv2.VideoCapture(1)

# Pasar path absoluto con video para usar uno ya grabado
# cap = cv2.VideoCapture(r"./pushup_video.mp4")


# Aca se crea una instancia del modelo de pose de MediaPipe con una confianza mínima para la detección y el seguimiento establecida en 0.7.
# Esto controla qué tan seguro debe estar el modelo para detectar y seguir los landmarks del cuerpo.
with mp_pose.Pose(min_detection_confidence=0.7,min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        #Se lee la imagen de la cámara (cap) y se redimensiona a 640x480 píxeles. Si la lectura no tiene éxito, se continúa el bucle.
        success, image = cap.read()
        image = cv2.resize(image, (640, 480))
        if not success:
            print("Ignorando el frame vacio de la camara.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        #La imagen se voltea horizontalmente (para dar una vista de tipo "selfie").
        #Se convierte de BGR (formato usado por OpenCV) a RGB (requerido por MediaPipe).
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # El modelo procesa la imagen para detectar los puntos clave (landmarks) del cuerpo.
        results = pose.process(image)

        #Dibujo de los landmarks y detección de flexiones
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lmList = findPosition(image, draw=True)
        if len(lmList) != 0:
            # Dibuja círculos en los landmarks del hombro derecho (lmList[12]) e izquierdo (lmList[11])
            cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 0, 255), cv2.FILLED)
            cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 0, 255), cv2.FILLED)
            if (lmList[12][2] and lmList[11][2] >= lmList[14][2] and lmList[13][2]):
                cv2.circle(image, (lmList[12][1], lmList[12][2]), 20, (0, 255, 0), cv2.FILLED)
                cv2.circle(image, (lmList[11][1], lmList[11][2]), 20, (0, 255, 0), cv2.FILLED)
                stage = "Abajo"
            if (lmList[12][2] and lmList[11][2] <= lmList[14][2] and lmList[13][2]) and stage == "Abajo":
                stage = "Arriba"
                counter += 1
                counter2 = str(int(counter))
                print(counter)
                # os.system("echo '" + counter2 + "' | festival --tts")
        text = "{}:{}".format("UAI - Contador de Push Ups", counter)
        cv2.putText(image, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        exit_text = "Presionar 'q' para salir. "

        # Obtener el tamaño de la imagen
        height, width, _ = image.shape

        # Calcular la posición para que el texto aparezca en la esquina inferior derecha
        text_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = width - text_size[0] - 10  # Restamos 10 para dejar un margen
        text_y = height - 10  # 10 píxeles arriba del borde inferior
        cv2.putText(image, exit_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('MediaPipe Pose', image)

        if create is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            create = cv2.VideoWriter(outputfile, fourcc, 30, (image.shape[1], image.shape[0]), True)
        create.write(image)
        key = cv2.waitKey(1) & 0xFF

        # Salir del programa cuando se presione la tecla "q"
        if key == ord("q"):
            break

# Eliminamos los objetos creados por cv2
cv2.destroyAllWindows()
