import math
import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
IMAGES_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
SAVE_PLOT = True
KEYS = ["pose.pose.x", "pose.pose.y", "pose.pose.theta"]

def rosbag_to_csv(data_base_path=None):
    csv_file_out_arr = []

    for trajectory in os.scandir(data_base_path):
        BAG_PATH = os.path.join(data_base_path, trajectory.name)

        if os.path.isfile(BAG_PATH):
            trajectory_bag = bagreader(BAG_PATH)

            for topic in trajectory_bag.topic_table['Topics']:
                filepath = trajectory_bag.message_by_topic(topic)  # save to csv by topic
                csv_file_out_arr.append(filepath)
                # print("rosbag_to_csv %s" % filepath)

    return csv_file_out_arr

def process_lidar_data(path_list):
    '''
    per ogni file "robot-front_laser-scan.csv" contenente i dati grezzi del lidar, crea un file
    chiamato "robot-processed_lidar.csv" con i seguenti campi: timestamp, LS (lidar sinistra), LC (lidar centro), LD (lidar destra)
    :param path_list: la lista di folders in cui c'è il file robot-front_laser-scan.csv
    :return: none
    '''

    for csv_filepath in path_list:
        filename = os.path.basename(csv_filepath)

        if not "front_laser" in filename:
            continue

        read_csv = pd.read_csv(csv_filepath, sep=",") # read existing file

        proc_data_dict = { # variable with processed data
            'Time': [],
            'LS': [],
            'LC': [],
            'LD': [],
        }

        for idx, row in read_csv.iterrows():
            if idx % 1000 == 0 or idx == 0:
                print("process row %d of %d" % (idx+1, read_csv.shape[0]))

            angle_min = row["angle_min"]
            angle_increment = row["angle_increment"]
            row_angles = angle_min + np.arange(541) * angle_increment
            # numpy.where(condition, [x, y, ]) # when condition True, yield x, otherwise yield y.
            row_angles = np.where(row_angles < 0, row_angles + 2*math.pi, row_angles)

            row_distances = row["ranges_0":"ranges_540"]
            angle_step = math.pi / 4

            # indici che soddisfano condizioni angoli dei 4 settori
            top_right_idx = np.argwhere(np.logical_and(row_angles >= 0, row_angles <= angle_step)) # da 0 a 45
            top_left_idx = np.argwhere(np.logical_and(row_angles >= (2*math.pi - angle_step), row_angles <= 2*math.pi)) # da 360-45 a 360
            right_idx = np.argwhere(np.logical_and(row_angles >= angle_step, row_angles <= math.pi / 2)) # da 45 a 90
            left_idx = np.argwhere(np.logical_and(row_angles >= (2*math.pi - math.pi / 2), row_angles <= (2*math.pi - angle_step))) # da 360-90 a 360-45
            # distanze corrispondenti agli indici trovati
            top_right_dist = row_distances[top_right_idx.flatten()]
            top_left_dist = row_distances[top_left_idx.flatten()]
            right_dist = row_distances[right_idx.flatten()]
            left_dist = row_distances[left_idx.flatten()]

            # calcola LC, LS, LR prendendo la minima distanza da ogni settore
            # questo è la logica che abbiamo usato in lab, ma può essere cambiata
            LC = min(min(min(top_left_dist), 10), min(min(top_right_dist), 10)) # centro
            LS = min(min(left_dist), 10) # sinistra
            LD = min(min(right_dist), 10) # destra

            # append to dict data
            proc_data_dict["Time"].append(row["Time"])
            proc_data_dict["LS"].append(LS)
            proc_data_dict["LC"].append(LC)
            proc_data_dict["LD"].append(LD)

        # save dict to new csv file
        output_filepath = csv_filepath.replace("front_laser-scan", "processed_lidar")
        df = pd.DataFrame(proc_data_dict)
        df.to_csv(output_filepath)

def plot_data(path_list=None, show_plot=False, save_plot=False):

    for csv_filepath in path_list:
        folder_name = csv_filepath.split("\\")[-1]
        folder_name = folder_name.split("/")[0]
        filename = os.path.basename(csv_filepath)

        if filename != "robot-robot_local_control-LocalizationComponent-status.csv":
            print("no", filename)
            continue
        else:
            print("ok", csv_filepath)

        csv_data = pd.read_csv(csv_filepath, sep=",")

        fig, axs = plt.subplots(3, 1, sharex=True)

        for j, key in enumerate(KEYS):
            # df = csv_df[key][i]
            csv_column = csv_data[f"{key}"]

            # axs[j].plot(df.index, df)
            axs[j].plot(csv_column.index, csv_column)
            axs[j].set_title(f"{key}")
            axs[j].legend([f"{folder_name}"])

        if save_plot:
            if not os.path.isdir(IMAGES_BASE_PATH):
                os.mkdir(IMAGES_BASE_PATH)

            image_path = os.path.join(IMAGES_BASE_PATH, f"{folder_name}.png")
            fig.savefig(image_path)

            plt.cla()

        if show_plot:
            plt.show()


if __name__ == '__main__':
    print("Export rosbag to csv")
    csv_file_out_arr = rosbag_to_csv(DATA_BASE_PATH)
    print(csv_file_out_arr)

    print("process lidar data")
    process_lidar_data(csv_file_out_arr)

    # print("plot 2d position")
    #plot_data(csv_file_out_arr, show_plot=False, save_plot=SAVE_PLOT)
