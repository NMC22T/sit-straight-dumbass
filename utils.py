# Utility Functions


# ------------------ Importing Libraries ------------------ #
from __future__ import annotations

import json
import numpy as np
import math
import tensorflow as tf
import os
import random
import playsound
import cv2

# ------------------ Importing Functions ------------------ #
from model import Model


# ------------------ Utility Functions ------------------ #
def open_thresholds(threshold_file: str = 'thresholds.json') -> dict:
    """
    open_thresholds: 
        Returns the cailbrated threshold files to be used

    Args:
        threshold_file (str): Name of the file containing the threshold values. Default is 'thresholds.json'.
        
    Returns:
        dict: A dictionary containing the threshold values for each joint keypoint.
    """

    try:
        with open(threshold_file, 'r') as file:
            thesholds = json.load(file)
        return thesholds
    except Exception as e:
        print(e)
        print("An error has occured. Ensure the theshold file is created by calibrating.")
        return None


def get_points_dictionary() -> dict:
    """
    get_points_dictionary: 
        Returns the mapped keypoint integer to each body part. Retrieved from TfHub.

    Returns:
        dict: A dictionary containing the numerical mappings of the keypoints.
    """

    return {
    "nose" : 0,
    "left_eye": 1,
    "right_eye" : 2,
    "left_ear" : 3,
    "right_ear" : 4,
    "left_shoulder" : 5,
    "right_shoulder" : 6,
    "left_elbow" : 7,
    "right_elbow" : 8,
    "left_wrist" : 9,
    "right_wrist" : 10,
    "left_hip" : 11,
    "right_hip" : 12,
    "left_knee" : 13,
    "right_knee" : 14,
    "left_ankle" : 15,
    "right_ankle" : 16
    }


def get_dist_between(frame: np.array, keypoints: np.array, p1_key: str, p2_key: str) -> float:
    """
    get_dist_between: 
        Determines the distance between two input points
    
    Args:
        frame (np.array): The current frame of the video feed
        keypoints (np.array): The current set of keypoints detected in the current frame
        p1_key (str): The key value of the first keypoint to calculate distance from
        p2_key (str): The key value of the second keypoint to calculate distance from

    Returns:
        dist (float): The distance between the two specified keypoints
    """

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    POINTS = get_points_dictionary()

    p1 = shaped[POINTS[p1_key]][:2].astype(int)
    p2 = shaped[POINTS[p2_key]][:2].astype(int)

    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    return dist


def get_dist_values(frame: np.array, keypoints: np.array) -> list[list, list, list, list, list, list]:
    """
    get_dist_values: 
        Returns a list of distances from different keypoints on the upper body

    Args:
        frame (np.array): The current frame being analyzed
        keypoints (np.array): The keypoints detected in the current frame
    
    Returns:
        (list): List of all distance values between keypoints
    """

    # Shoulder to Ear
    dists_right_ear_dist = get_dist_between(frame, keypoints, "right_shoulder", 'right_ear')
    dists_left_ear_dist = get_dist_between(frame, keypoints, "left_shoulder", 'left_ear')

    # Shoulder to Nose
    dists_right_nose_dist = get_dist_between(frame, keypoints, "right_shoulder", "nose")
    dists_left_nose_dist = get_dist_between(frame, keypoints, "left_shoulder", "nose")

    # Shoulder to Eyes
    dists_right_eyes_dist = get_dist_between(frame, keypoints, "right_shoulder", "right_eye")
    dists_left_eyes_dist = get_dist_between(frame, keypoints, "left_shoulder", "left_eye")

    return [dists_right_ear_dist, dists_left_ear_dist, dists_right_nose_dist, dists_left_nose_dist, dists_right_eyes_dist, dists_left_eyes_dist]


def reshape_image(frame: np.array, model: Model) -> tf.Tensor:
    """
    reshape_image: 
        Reshaping the camera input frame to fit the model

    Args:
        frame (np.array): A NumPy array representing the input image frame.
        model (Model): A machine learning model with an expected input shape.

    Returns:
        tf.Tensor: A TensorFlow tensor representing the resized image, with data type tf.float32.
    """
    
    image = frame.copy()
    image = tf.image.resize_with_pad( np.expand_dims(image, axis=0), model.input_dim[0], model.input_dim[1] )
    input_image = tf.cast(image, dtype=tf.float32)

    return input_image


def input_output_details(interpreter: tf.lite.Interpreter) -> tuple[list, list]:
    """
    input_output_details: 
        Used to get the details from the interperter

    Args:
        interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter to get input and output details from.

    Returns:
        tuple: A tuple of two lists containing input and output details respectively.
    """
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return input_details, output_details


def make_prediction(interpreter: tf.lite.Interpreter, input_details: list, output_details:list, input_image: np.array) -> np.array:
    """
    make_prediction: 
        Used to get the keypoints output from the provided image

    Args:
        interpreter: A TensorFlow Lite interpreter object.
        input_details: A list of input details, as returned by `interpreter.get_input_details()`.
        output_details: A list of output details, as returned by `interpreter.get_output_details()`.
        input_image: A NumPy array representing the input image to use for inference.

    Returns:
        np.array: A NumPy array containing the predicted keypoints with confidence scores.
    """

    interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
    interpreter.invoke()
    keypoints_with_score = interpreter.get_tensor(output_details[0]['index'])

    return keypoints_with_score


def get_audio_list(filepath: str) -> list[str]:
    """
    get_audio_list: 
        Used to create a list of filepaths for all available audio recordings

    Parameters:
        filepath (str): The path to the directory where the audio recordings are stored.

    Returns:
        list[str]: A list of file paths for all available audio recordings in the directory.
    """

    return [file.path for file in os.scandir(filepath)]


def play_audio_recording(audio_list: list) -> None:
    """
    play_audio_recording: 
        An event trigger for when posture is bad longer than theshold. Plays pre-recorded audio files.

    Parameters:
        audio_list (list): A list of file paths for audio recordings to play.
    """

    audio_to_play = random.choice(audio_list)
    playsound.playsound(audio_to_play)


def model_name_input() -> str:
    """
    model_name_input:
        Asks which model to use until until a valid model is provided.
    
    Returns:
        str: The name of the selected model.
    """
    
    list_of_models = ["thunder", "lightning"]

    while True:
        model_name = input("What is the model you want to use? lightning fast but bad, thunder slow but good:\n")

        if model_name in list_of_models:
            return str(model_name)
        
        print("Try again, not a valid model\n")


def calibration_input() -> bool:
    """
    calibration_input:
        Used to determine whether to run the calibration or not.
    
    Returns:
        bool: True if calibration needs to be run, False otherwise.
    """

    if not os.path.exists('thresholds.json'):
        print("No calibration file exsists, running calibration\n")
        return True
    
    run_calibration = input("Do you want to run the calibration? Type yes otherwise defaults to no:\n")
    return (run_calibration == "yes")


def keypoint_initialization(capture_front: cv2.VideoCapture, model: Model, interpretor: tf.lite.Interpreter) -> tuple(np.array, np.array):
    """
    keypoint_initialization:
        Used to initial the openCV frame and return the detected keypoints.

    Args:
        capture_front (cv2.VideoCapture): The video capture object to read frames from.
        model (Model): The pre-trained model to use for keypoint detection.
        interpretor (tf.lite.Interpreter): The interpreter to run the model with.

    Returns:
    A tuple of two numpy arrays:
        frame_front (np.array): The OpenCV frame obtained from the video capture object.
        keypoint_score_front (np.array): A 2D array of shape (N, 4) representing the detected keypoints and their scores,
            where N is the number of keypoints and each keypoint is represented by 4 values (x, y, score, class).
    """

    # Read Camera Input
    ret_front, frame_front = capture_front.read()

    # Image Reshape
    input_image_front = reshape_image(frame=frame_front, model=model)

    # Setup Tensor Input and Output
    input_details, output_details = input_output_details(interpreter=interpretor)

    # Make Prediction
    keypoint_score_front = make_prediction(interpreter=interpretor, input_details=input_details, output_details=output_details, input_image=input_image_front)

    return frame_front, keypoint_score_front