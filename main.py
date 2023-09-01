import math
import bagpy
from bagpy import bagreader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

###
# per il dataset:
# fai una cartella "dataset" e dentro ci metti i file .bag
###

DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset')
IMAGES_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'images')
SAVE_PLOT = True
KEYS = ["pose.pose.x", "pose.pose.y", "pose.pose.theta"]

def rosbag_to_csv(data_base_path=None):
    csv_path_list = [] # lista con path di tutti i csv creati
    dir_path_list = [] # lista con path di tutte le cartelle create

    # converti bag in csv
    for entry in os.scandir(data_base_path):
        path = os.path.join(data_base_path, entry.name)

        if os.path.isfile(path):
            trajectory_bag = bagreader(path)

            for topic in trajectory_bag.topic_table['Topics']:
                filepath = trajectory_bag.message_by_topic(topic)  # save to csv by topic
                csv_path_list.append(filepath)
                # print("rosbag_to_csv %s" % filepath)

    # popola array con lista di cartelle create (later use)
    for entry in os.scandir(data_base_path):
        path = os.path.join(data_base_path, entry.name)

        if os.path.isdir(path):
            dir_path_list.append(path)

    return [csv_path_list, dir_path_list]

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

def create_ml_csv(dir_path_list):
    '''
    per ogni cartella crea un file "ml_data_xxx.csv" contenente i dati processati del lidar (LS, LC, LD) e i dati della posa del robot (x, y, theta)
    questo sarà il file che viene dato in pasto al modello machine learning, hence il nome "ml_data"
    :param path_list: la lista di folders in cui sono presenti il file con i dati lidar processati (robot_processed_lidar.csv)
    e il file con i dati della posa del robot (robot-robot_local_control-LocalizationComponent-status.csv)
    :return: none
    '''

    for curr_dir in dir_path_list:
        print("processing folder: %s" % curr_dir)

        combined_data = {  # sia dati lidar (LS, LC, LD) che dati posizione (x, y, theta)
            'Time_lidar': [],
            'Time_pose': [],
            'x': [],
            'y': [],
            'theta': [],
            'LS': [],
            'LC': [],
            'LD': [],
        }

        for entry in os.scandir(curr_dir):
            path = os.path.join(curr_dir, entry.name)

            if not ("processed_lidar" in path or "LocalizationComponent" in path):
                continue

            if "processed_lidar" in path:
                csv_data = pd.read_csv(path, sep=",")  # read existing file
                combined_data["Time_lidar"] = csv_data["Time"]
                combined_data["LS"] = csv_data["LS"]
                combined_data["LC"] = csv_data["LC"]
                combined_data["LD"] = csv_data["LD"]

            if "LocalizationComponent" in path:
                csv_data = pd.read_csv(path, sep=",")  # read existing file
                combined_data["Time_pose"] = csv_data["Time"]
                combined_data["x"] = csv_data["pose.pose.x"]
                combined_data["y"] = csv_data["pose.pose.y"]
                combined_data["theta"] = csv_data["pose.pose.theta"]

        if len(combined_data["Time_lidar"]) > 0:
            # re-arrange data perchè i dati pose sono inviati con una frequenza minore,
            # quindi ho molti piu dati lidar che dati pose
            # quindi non posso semplicemente affiancare le due colonne di dati altrimenti la colonna lidar è lunga 1000 e quella pose 100
            new_x = [-1] * len(combined_data["Time_lidar"])
            new_y = [-1] * len(combined_data["Time_lidar"])
            new_theta = [-1] * len(combined_data["Time_lidar"])
            new_time = [-1] * len(combined_data["Time_lidar"])

            for i in range(len(combined_data["Time_pose"])):
                # trova time instant lidar piu vicino a time instant pose
                curr_pose_time = combined_data["Time_pose"][i]
                up_limit = curr_pose_time + 0.1
                bottom_limit = curr_pose_time - 0.1
                time_idx = np.argwhere(np.logical_and(combined_data["Time_lidar"] >= bottom_limit, combined_data["Time_lidar"] <= up_limit))
                time_idx = time_idx.flatten()[0]

                # sposta dati pose all'indice trovato
                new_x[time_idx] = combined_data["x"][i]
                new_y[time_idx] = combined_data["y"][i]
                new_theta[time_idx] = combined_data["theta"][i]
                new_time[time_idx] = combined_data["Time_pose"][i]
                # print("pose %.5f moved to lidar %.5f" % (curr_pose_time, combined_data["Time_lidar"][time_idx]))

            combined_data["x"] = new_x
            combined_data["y"] = new_y
            combined_data["theta"] = new_theta
            combined_data["Time_pose"] = new_time


            # save to file
            dirname = os.path.basename(curr_dir)
            output_filename = "ml_data_" + dirname + ".csv"
            output_filepath = os.path.join(curr_dir, output_filename)
            df = pd.DataFrame(combined_data)
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
    [csv_path_list, dir_path_list] = rosbag_to_csv(DATA_BASE_PATH)

    print(csv_path_list)
    print(dir_path_list)

    print("process lidar data")
    process_lidar_data(csv_path_list)

    create_ml_csv(dir_path_list)

    # print("plot 2d position")
    #plot_data(csv_path_list, show_plot=False, save_plot=SAVE_PLOT)
