import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import time
import pandas as pd

model = load_model('smnist.h5')

mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Specify the path to your video file
video_file_path = 'your_video.mp4'
cap = cv2.VideoCapture(video_file_path)

_, frame = cap.read()

h, w, c = frame.shape

img_counter = 0
analysisframe = ''
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Create a text file to save recognized characters
output_file = open("recognized_characters.txt", "w")

# Define the frame interval (e.g., every 24 frames for 24fps video)
frame_interval = 1
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        # End of video
        break

    frame_count += 1

    analysisframe = frame
    showframe = analysisframe

    # Check if the frame is not in BGR format
    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0: 
        if frame.shape[2] == 1:  # Assuming it's grayscale (1 channel)
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:  # Assuming it's RGBA (4 channels)
            analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_RGBA2BGR)

        cv2.imshow("Frame", showframe)
        framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
        resultanalysis = hands.process(framergbanalysis)
        hand_landmarksanalysis = resultanalysis.multi_hand_landmarks

        # The rest of your analysis code ...

        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        hand_landmarks = result.multi_hand_landmarks

        if frame_count % frame_interval == 0:
            if hand_landmarksanalysis:
                for handLMsanalysis in hand_landmarksanalysis:
                    x_max = 0
                    y_max = 0
                    x_min = w
                    y_min = h
                    for lmanalysis in handLMsanalysis.landmark:
                        x, y = int(lmanalysis.x * w), int(lmanalysis.y * h)
                        if x > x_max:
                            x_max = x
                        if x < x_min:
                            x_min = x
                        if y > y_max:
                            y_max = y
                        if y < y_min:
                            y_min = y
                    y_min -= 20
                    y_max += 20
                    x_min -= 20
                    x_max += 20

                analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe, (28, 28))

                nlist = []
                rows, cols = analysisframe.shape
                for i in range(rows):
                    for j in range(cols):
                        k = analysisframe[i, j]
                        nlist.append(k)

                datan = pd.DataFrame(nlist).T
                colname = []
                for val in range(784):
                    colname.append(val)
                datan.columns = colname

                pixeldata = datan.values
                pixeldata = pixeldata / 255
                pixeldata = pixeldata.reshape(-1, 28, 28, 1)
                prediction = model.predict(pixeldata)
                predarray = np.array(prediction[0])
                letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                predarrayordered = sorted(predarray, reverse=True)
                high1 = predarrayordered[0]
                high2 = predarrayordered[1]
                high3 = predarrayordered[2]

                recognized_characters = []

                for key, value in letter_prediction_dict.items():
                    if value == high1:
                        recognized_characters.append((key, 100 * value))
                        print("Predicted Character 1: ", key)
                        print('Confidence 1: ', 100 * value)

                # Write recognized characters to the output file
                for char, confidence in recognized_characters:
                    output_file.write(f"Predicted Character: {char}, Confidence: {confidence}\n")

    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0: 
        cv2.imshow("Frame", frame)

cap.release()
output_file.close()
cv2.destroyAllWindows()
