# Calibartor Functions


# ------------------ Importing Libraries ------------------ #
from __future__ import annotations

from scipy.stats import trim_mean
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import json

from model import Model
import tensorflow as tf


# ------------------ Importing Functions ------------------ #
from utils import get_dist_values, keypoint_initialization
from debug import keypoint_renderings


# ------------------ Calibrator Functions ------------------ #
def calibrate(model: Model, interpretor: tf.lite.Interpreter, calibration_time: int = 30) -> dict[str, float]:
    """
    calibrate: 
        Runs the calibration for each user. Saves a thresholds.json file for output.

    Args:
        model (Model): A machine learning model used for pose detection.
        interpretor (tf.lite.Interpreter): An interpreter for the given machine learning model.
        calibration_time (int): The amount of time in seconds to spend calibrating each posture.
    """

    def find_trimmed_mean(input_list: list, trim_percent: float) -> list:
        """
        find_trimmed_mean: 
            Returns the trimmed mean for a list of input data

        Args:
            input_list (list): A list of data distributions, each represented as a list of numbers.
            trim_percent (float): The percentage of data to trim from both ends of each distribution,
                expressed as a decimal between 0 and 0.5.

        Returns:
            list: A list of trimmed means, one for each distribution in the input list.
        """

        return [trim_mean(dist, trim_percent) for dist in input_list]

        # mean_list = []
        # for dist in input_list:
        #     mean_list.append(trim_mean(dist, trim_percent))

        # return mean_list
    

    def get_dist_lst_values(input_list: list, frame: np.array, keypoints: np.array) -> list:
        """
        get_dist_lst_values: 
            Adds distance values for each frame in a list and returns a list of the lists.

        Args:
            input_list (list): A list of lists, where each inner list corresponds to a particular frame and contains a set of values.
            frame (np.ndarray): A numpy array containing the pixel values for a single frame.
            keypoints (list): A list of tuples, where each tuple contains the (x,y) coordinates of a keypoint on the frame.

        Returns:
            list: A list of lists, where each inner list corresponds to a particular frame and contains a set of values, including the newly appended distance values.
        """

        new_dist_values = get_dist_values(frame, keypoints)
        for i in range(len(input_list)):
            input_list[i].append(new_dist_values[i])

        return input_list


    def calibrator_video() -> tuple[list, list]:
        """
        calibrator_video: 
            Runs the opencv video for the calibration data to be recorded.

        Args:
            input_list (list): A list of lists, where each inner list corresponds to a particular frame and contains a set of values.
            frame (np.ndarray): A numpy array containing the pixel values for a single frame.
            keypoints (list): A list of tuples, where each tuple contains the (x,y) coordinates of a keypoint on the frame.

        Returns:
            list: A list of lists, where each inner list corresponds to a particular frame and contains a set of values, including the newly appended distance values.
        """

        capture_front = cv2.VideoCapture(0)
        start_time = time.perf_counter()

        current_calibration_time = 0
        
        good_calibration_list = [[], [], [], [], [], []]
        bad_calibration_list = [[], [], [], [], [], []]

        while capture_front.isOpened():
            
            frame_front, keypoint_score_front = keypoint_initialization(capture_front, model, interpretor)
            
            confidence_threshold = 0.4
            keypoint_renderings(frame_front, keypoint_score_front, confidence_threshold)

            if calibration_time > current_calibration_time:
                # Calibrate Good Posture
                cv2.putText(frame_front, 'Calibrating Good Posture : ' + str(int(calibration_time - current_calibration_time)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                good_calibration_list = get_dist_lst_values(good_calibration_list, frame_front, keypoint_score_front)
            elif (2*calibration_time) > current_calibration_time:
                # Calibrate Bad Posture
                cv2.putText(frame_front, 'Calibrating Bad Posture : '  + str(int(2*calibration_time - current_calibration_time)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                bad_calibration_list = get_dist_lst_values(bad_calibration_list, frame_front, keypoint_score_front)
            else:
                break

            current_calibration_time = time.perf_counter() - start_time
            cv2.imshow("Front", frame_front)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        capture_front.release()
        cv2.destroyAllWindows()

        return good_calibration_list, bad_calibration_list


    def view_calibartion(thresholds: list[float], jump_percent: float, raw_good: list[list[float]], raw_bad: list[list[float]], trimmed_good: list[float], trimmed_bad: list[float]) -> None:
        """
        view_calibration: 
            Shows the results of the calibration using graphs.

        Args:
            thresholds (list[float]): A list of threshold values used for the calibration.
            jump_percent (float): The percentage used to calculate the jump threshold.
            raw_good (list[list[float]]): A list of lists of raw values for the good class, where each inner list corresponds to a particular threshold.
            raw_bad (list[list[float]]): A list of lists of raw values for the bad class, where each inner list corresponds to a particular threshold.
            trimmed_good (list[float]): A list of trimmed mean values for the good class, where each value corresponds to a particular threshold.
            trimmed_bad (list[float]): A list of trimmed mean values for the bad class, where each value corresponds to a particular threshold.
        """

        fig, axs = plt.subplots(2, 3)
        fig.set_size_inches(18.5, 10.5)

        # Graphs and stuff
        for i, threshold in enumerate(thresholds):
            axs[int(i%2),int(i/2)].set_title(threshold)
            axs[int(i%2),int(i/2)].plot(raw_good[i] + raw_bad[i], color='orange')
            axs[int(i%2),int(i/2)].axhline(trimmed_good[i], color='green', xmin=0, xmax=len(raw_good[i])/len(raw_good[i]+raw_bad[i]))
            axs[int(i%2),int(i/2)].axhline(trimmed_bad[i], color='red', xmin=len(raw_good[i])/len(raw_good[i]+raw_bad[i]), xmax=1)
            axs[int(i%2),int(i/2)].axhline(trimmed_bad[i] + jump_percent*(trimmed_good[i]-trimmed_bad[i]), color='blue')


    def run_calibration(trimmed_percent: float = 0.1, jump_percent: float = 0.5, debug: bool = False) -> dict[str, float]:
        """
        run_calibration: 
            Used to calculate the threshold values from the recorded calibration data.

        Args:
            trimmed_percent (float): The percentage of data to trim from both ends of the calibration data.
            jump_percent (float): The percentage to jump from the bad trimmed mean to get the threshold value.
            debug (bool): Whether to display calibration graphs for debugging purposes.

        Returns:
            dict: A dictionary containing the calculated threshold values for each keypoint.
        """

        # Finding Threshold
        thresholds = {
            'dists_right_ear' : 0,
            'dists_left_ear' : 0,
            'dists_right_nose' : 0,
            'dists_left_nose' : 0,
            'dists_right_eye' : 0,
            'dists_left_eye' : 0
        }

        raw_good, raw_bad = calibrator_video()

        # Using trimmed means for threshold Values
        trimmed_good = find_trimmed_mean(raw_good, trimmed_percent)
        trimmed_bad = find_trimmed_mean(raw_bad, trimmed_percent)

        for i, threshold_key in enumerate(thresholds):
            thresholds[threshold_key] = trimmed_bad[i] + jump_percent * (trimmed_good[i] - trimmed_bad[i])

        if debug:
            view_calibartion(thresholds, jump_percent, raw_good, raw_bad, trimmed_good, trimmed_bad)
        
        return thresholds

    
    return run_calibration(trimmed_percent=0.1, jump_percent=0.5, debug=False)


def save_thresholds(thresholds: dict[str, float]) -> None:
    """
    save_thresholds: 
        Saves the theshold.json to be used after inital calibration

    Args:
        thresholds (dict): A dictionary containing the threshold values.
    """

    with open("thresholds.json", 'w') as threshold_file:
        json.dump(thresholds, threshold_file)