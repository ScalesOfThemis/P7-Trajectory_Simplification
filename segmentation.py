import os
import sys
from time import time
from  tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_smoothing_spline, BSpline
from scipy.signal import find_peaks
from sys import getsizeof
from math import sqrt
# from curve_fitter import fit_polynomial
np.set_printoptions(threshold=np.inf)

lon_segments = []
lat_segments = []
time_segments = []

path = '../P8-Trajectory_Simplification/'

def segment_trajectories(T_num=1000):

    dir = f'{path}Geolife_out'
    print(dir)
    # I think I am doing something wrong in terms of what trajs contains, as it does not have the same length as traj_list, will look into it later
    traj_list = sorted(os.listdir(dir))  
    trajs = []
    cnt = 0

    for traj_filename in tqdm(traj_list, desc='Loading Trajectories'):


        filename = dir + '/' + traj_filename

        with open(filename, 'r') as f:

            lines = f.readlines()
            if len(lines) >=T_num:
                for i in range(len(lines)//T_num):
                    lon = []
                    lat = []
                    time = []
                    for line in lines[i*T_num:i*T_num+T_num]:
                        temp = line.strip().split(' ')
                        if len(temp) < 3:
                            continue
                        lon.append(float(temp[1]))
                        lat.append(float(temp[0]))
                        time.append(int(temp[2]))
                        # traj.append([lon, lat, time])
                    trajs.append([lon, lat, time])
                cnt += len(lines)//T_num 

    # create_original_graph(trajs, traj_list)


    trajs_segmented = []

    for i in [0]: #tqdm(range(len(trajs)), desc='Segmenting trajectories'):
        lons, lats, times = segment_trajectory(trajs[i][0], trajs[i][1], trajs[i][2])
        trajs_segmented.append([lons.copy(), lats.copy(), times.copy()])
        lon_segments.clear()
        lat_segments.clear()
        time_segments.clear()

    # create_fit_graph(trajs_segmented,traj_list)


def create_original_graph(trajs, names):
    dir = f'{path}Original_trajectories'
    if not os.path.exists(dir):
        os.makedirs(dir)

    print(len(trajs))
    print(len(names))
    
    for traj, name in zip(trajs, names):
        lon = traj[0]
        lat = traj[1]
        plt.figure(figsize=(8, 6))
        plt.plot(
            lon, 
            lat, 
            color='blue', 
            marker='o'
        )
        plt.title(f'Trajectory {name}')
        plt.xlabel('lon')
        plt.ylabel('lat')

        plt.savefig(f'{path}Original_trajectories/{name}')
        plt.close()

def create_fit_graph(trajs, name):
    dir = f'{path}Simplified_trajectories'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for traj, name in zip(trajs, name):

        lons = traj[0]
        lats = traj[1]
        times = traj[2]

        plt.figure(figsize=(8, 6))

        for lon, lat, time in zip(lons, lats, times):
            lon_polynomial = fit_polynomial(time, lon)
            lat_polynomial = fit_polynomial(time, lat)
            plt.plot(
                lat_polynomial((np.linspace(time[0], time[-1], len(time)*10))), 
                lon_polynomial((np.linspace(time[0], time[-1], len(time)*10))), 
                color='red', 
            )
        plt.title(f'Polynomial prediction of trajectory {name}')
        plt.xlabel('lat')
        plt.ylabel('lon')

        plt.savefig(f'{path}Simplified_trajectories/{name}')
        plt.close()
    return lon_polynomial, lat_polynomial


def fit_polynomial(x,y):
    max_deg = len(x)*2//3
    max_error = abs(max(y) - min(y)) * 0.1
    for i in range(max_deg):
        polynomial = np.polynomial.Polynomial.fit(x, y, deg=i, full=True)
        if polynomial[1][0].size < 1 or polynomial[1][0] < max_error: 
            # print(f"Best fit polynomial degree: {i}")
            return polynomial[0]
    # print(f"No suitable polynomial found within degree 0 to {max_deg}.")
    return polynomial[0] 



def segment_trajectory(lon, lat, time):

    # curvatures = calculate_curvature(lat, lon, time)

    # peaks, properties = find_peaks(
    # curvatures,
    # prominence=1000,
    # distance=5
    # )

    # indices = peaks

    # curvature_mean = sum(curvatures)/len(curvatures)

    # print(curvature_mean)

    # print(type(curvatures))
    # indices = [i for i, val in enumerate(curvatures) if abs(val) >= curvature_mean]


    turning_angles = calculate_turning_angles(lat, lon)

    
    peaks, properties = find_peaks(
    turning_angles,
    prominence=15,
    distance=1
    )

    indices = peaks

    plt.figure()
    plt.plot(turning_angles)
    plt.show
    plt.close


    plt.figure()
    plt.plot(
        # lat_spline.__call__(time),
        # lon_spline.__call__(time)
        lat,
        lon,
        'bo'
    )
    for i in indices :
        plt.plot(
            lat[i],
            lon[i],
            'ro'
        )
    plt.show()
    plt.close()

    return [], [], []

    
    # if segment_end < segment_upper_bound:
    #     return segment_trajectory(lon, lat, time, segment_start, segment_end + max((segment_upper_bound - segment_end) // 2, 1), segment_upper_bound)
    # elif end_criterion:
    #     lon_segments.append(lon[segment_start:segment_end])
    #     lat_segments.append(lat[segment_start:segment_end])
    #     time_segments.append(time[segment_start:segment_end])
    #     if segment_end < len(lon):
    #         return segment_trajectory(lon, lat, time, segment_end)
    #     return lon_segments, lat_segments, time_segments
    # else :
    #     segment_end = segment_start + max((segment_end - segment_start) // 2, 1)
    #     segment_upper_bound = segment_end
    #     return segment_trajectory(lon, lat, time, segment_start, segment_end, segment_upper_bound)


def calculate_curvature(x, y, t):

    x_spline = make_smoothing_spline(t, x)
    y_spline = make_smoothing_spline(t, y)

    dx = x_spline.derivative(1)(t)
    dy = y_spline.derivative(1)(t)

    ddx = x_spline.derivative(2)(t)
    ddy = y_spline.derivative(2)(t)

    cross_product = np.abs(dx * ddy - ddx * dy)

    speed = np.hypot(dx, dy)

    plt.figure(figsize=(12, 4))
    plt.plot(speed)
    plt.title("Speed along the curve")
    plt.show()

    # Set a lower limit on the speed, as low speed will lead to abnormally high curvature
    epsilon = 1e-8
    curvature = cross_product / np.maximum(speed, epsilon)**3

    return curvature

def calculate_turning_angles(x, y):

    points = np.column_stack((x, y))

    v1 = points[1:-1] - points[:-2]
    v2 = points[2:] - points[1:-1]

    dot = np.sum(v1 * v2, axis=1)

    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)

    denominator = norm1 * norm2

    cos_angle = dot / np.maximum(denominator, 1e-12)

    angles = np.arccos(np.clip(cos_angle, -1, 1))

    return np.degrees(angles)
        


segment_trajectories()