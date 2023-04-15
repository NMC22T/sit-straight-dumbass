# Detector Functions


# ------------------ Importing Libraries ------------------ #
import cv2
import os
import time
import tensorflow as tf 


# ------------------ Importing Functions ------------------ #
from model import Model
from utils import keypoint_initialization, open_thresholds, get_audio_list, get_dist_values, play_audio_recording
from debug import keypoint_renderings


# ------------------ Detector Function ------------------ #
def detector(model: Model, interpretor: tf.lite.Interpreter, debug: bool = False) -> None:
    """
    detector:
        Main function that operates the detection and event trigger for the application.

    Parameters:
        model (TensorFlow model): The TensorFlow model to use for detecting keypoint scores.
        interpreter (TFLiteInterpreter): The TFLite interpreter to use for detecting keypoint scores.
        debug (bool): Whether to show debug information.
    """

    capture_front = cv2.VideoCapture(0)

    time_threshold = 15
    dist_thresholds = open_thresholds()

    basepath = os.getcwd()

    audio_filepath = os.path.join(basepath, 'audio_files')
    audio_list = get_audio_list(audio_filepath)
    playing_audio = False

    time_count = 0
    start_time = time.time() 

    while capture_front.isOpened():

        # Get the front frame and keypoint score
        frame_front, keypoint_score_front = keypoint_initialization(capture_front, model, interpretor)

        # Show the frame with keypoints if debug mode is on
        if debug:
            confidence_threshold=0.4
            keypoint_renderings(frame_front, keypoint_score_front, confidence_threshold)

        # Determine Distances
        current_distances = get_dist_values(frame=frame_front, keypoints=keypoint_score_front)
        
        
        # If all distances are above threshold, then posture is correct. Else, posture is not correct.
        if all([current_distances[i] > dist_thresholds[threshold] for i, threshold in enumerate(dist_thresholds)]):
            time_count = 0
            start_time = time.time()
        else:
            if not playing_audio:
                current_time = time.time()
                time_count += current_time - start_time
                start_time = current_time
            else:
                time_count = 0
                start_time = time.time()

            # If time threshold has been reached, play audio recording
            if (time_count > time_threshold):
                playing_audio = True
                time_count = 0
                start_time = time.time()

                play_audio_recording(audio_list)

                playing_audio = False
                time_count = 0
                start_time = time.time()
        
        # Show the front frame if debug mode is on
        if debug:
            print(int(time_count))
            cv2.imshow("Front", frame_front)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture_front.release()
    cv2.destroyAllWindows()