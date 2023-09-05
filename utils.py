import os

DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
IMAGES_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
SAVE_PLOT = True
KEYS = ["pose.pose.x", "pose.pose.y", "pose.pose.theta"]