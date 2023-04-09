# Debug Functions


# ------------------ Importing Libraries ------------------ #
import numpy as np
import cv2


# ------------------ Drawing Utilities ------------------ #
def draw_keypoints(frame: np.array, keypoints: np.array, confidence_threshold: float) -> None:
    """
    draw_keypoint: 
        Used to draw the keypoint outputs. Used only when debugging or calibrating.

    Parameters:
        frame (numpy.ndarray): The input frame to draw keypoints on.
        keypoints (numpy.ndarray): The keypoints to draw, in the format (y, x, confidence).
        confidence_threshold (float): The minimum confidence score required for a keypoint to be drawn.

    Returns:
        None
    """

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    # Group keypoints with confidence scores above the threshold
    for kp in shaped:
        ky, kx, conf = kp
        if conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)


def draw_connections(frame: np.array, keypoints: np.array, edges: dict, confidence_threshold: float = 0.5) -> None:
    """
    draw_connections: 
        Used to draw the edges between keypoint outputs. Used only when debugging or calibrating.

     Parameters:
        frame (numpy.ndarray): The input frame to draw edges on.
        keypoints (numpy.ndarray): The keypoints to use for drawing edges, in the format (y, x, confidence).
        edges (dict): A dictionary where keys are tuples of the form (p1, p2) representing the indices of keypoints to 
                      connect and values are tuples of the form (r, g, b) representing the color of the line to draw.
        confidence_threshold (float, optional): The minimum confidence score required for a keypoint to be used in drawing
                                                 an edge. Defaults to 0.5.

    Returns:
        None
    """

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if min(c1, c2) > confidence_threshold:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 1)


def get_edge_dictionary() -> dict:
    """
    get_edge_dictionary: 
        Used to map pairs of keypoints to create an edge
    
    Returns:
        dict: A dictionary that maps pairs of keypoints to create an edge.
    """

    return {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
    }


def keypoint_renderings(frame_front: np.array, keypoint_score_front: np.array, confidence_threshold: float = 0.4) -> None:
    """
    keypoint_renderings: 
        Used to render keypoints and edges on the openCV window.

    Parameters:
        frame_front (numpy.ndarray): The input frame to draw keypoints on.
        keypoint_score_front (numpy.ndarray): The keypoints to draw, in the format (y, x, confidence).
        confidence_threshold (float): The minimum confidence score required for a keypoint to be drawn. Default is 0.4.

    Returns:
        None
    """

    # Rendering Points
    draw_keypoints(frame=frame_front, keypoints=keypoint_score_front, confidence_threshold=0.4)

    # Rendering Edges
    EDGES = get_edge_dictionary()
    draw_connections(frame=frame_front, keypoints=keypoint_score_front, edges=EDGES, confidence_threshold=confidence_threshold)