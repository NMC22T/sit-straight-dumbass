# Sit Straight
# NMC22T

# ------------------ Importing Libraries ------------------ #
import tensorflow as tf


# ------------------ Importing Functions ------------------ #
from model import Model
from calibrator import calibrate, save_thresholds
from detector import detector
from utils import model_name_input, calibration_input


# ------------------ Main Function ------------------ #
def main():
    model_name = model_name_input()
    run_calibration = calibration_input()


    model = Model(name=model_name)
    model.download_model()

    interpretor = tf.lite.Interpreter(model_path=model.file_path)
    interpretor.allocate_tensors()


    if run_calibration:
        print("Running Calibration\n")
        save_thresholds(calibrate(model=model, interpretor=interpretor, calibration_time=15))
    
    detector(model=model, interpretor=interpretor, debug=False)


# ------------------ Start Defenition ------------------ #
if __name__ == '__main__':
    main()