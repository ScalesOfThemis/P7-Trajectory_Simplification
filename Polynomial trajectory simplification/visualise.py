import os

import sys
from time import time
from  tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_smoothing_spline, BSpline
from sys import getsizeof
# from curve_fitter import fit_polynomial
np.set_printoptions(threshold=np.inf)


lon_segments = []
lat_segments = []
time_segments = []

N = 10 #Lower bound on segment size

path = '../Polynomial trajectory simplification/'

def create_visualisation(T_num=1000):

    dir = f'{path}Geolife_out'
    print(dir)
    # I think I am doing something wrong in terms of what trajs contains, as it does not have the same length as traj_list, will look into it later
    traj_list = sorted(os.listdir(dir))  
    trajs = []
    cnt = 0

    for traj_filename in tqdm(traj_list, desc='Loading Trajectories'):

        
        filename = dir + '/' + traj_filename
        #print(filename)
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

    create_original_graph(trajs, traj_list)

    
    trajs_segmented = []

    for i in tqdm(range(len(trajs)), desc='Segmenting trajectories'):
        lons, lats, times = segment_trajectory(trajs[i][0], trajs[i][1], trajs[i][2])
        trajs_segmented.append([lons.copy(), lats.copy(), times.copy()])
        lon_segments.clear()
        lat_segments.clear()
        time_segments.clear()

    create_fit_graph(trajs_segmented, traj_list)
    
    trajs_segmented = []    

    for i in tqdm(range(len(trajs)), desc='Segmenting trajectories'):
        trajs_segmented.append(segment_trajectory_dyn(trajs[i][0], trajs[i][1], trajs[i][2]))

    create_fit_graph_new(trajs_segmented,traj_list)
    

def create_fit_graph_new(trajs, name):
    dir = f'{path}Simplified_trajectories_new'
    if not os.path.exists(dir):
        os.makedirs(dir)


    for traj, name in zip(trajs, name):
        lons = []
        
        lats = []
        
        #time = [x for x in range(0,traj[1][-1][1])]

        for polynomial, segment in zip(traj[0],traj[1]):
            for i in range(segment[0],segment[1] + 1):
                lats.append(polynomial(i))

        for polynomial, segment in zip(traj[2],traj[3]):
            for i in range(segment[0],segment[1] + 1):
                lons.append(polynomial(i))
        
        plt.figure(figsize=(8, 6))

        plt.plot(lons,lats)

        plt.title(f'Polynomial prediction of trajectory {name}')
        plt.xlabel('lon')
        plt.ylabel('lat')

        plt.savefig(f'{path}Simplified_trajectories_new/{name}')
        plt.close()


def create_original_graph(trajs, names):
    dir = f'{path}Original_trajectories'
    if not os.path.exists(dir):
        os.makedirs(dir)

    #print(len(trajs))
    #print(len(names))
    
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
                lon_polynomial((np.linspace(time[0], time[-1], len(time)*10))), 
                lat_polynomial((np.linspace(time[0], time[-1], len(time)*10))), 
                color='red', 
            )
        plt.title(f'Polynomial prediction of trajectory {name}')
        plt.xlabel('lon')
        plt.ylabel('lat')

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

def segment_trajectory_dyn(lon, lat, time, segment_upper_bound=None):
    pre_semgentation = find_jumps_(lon, lat)
    polynomials_lat = [] 
    lat_segments = []
    polynomials_lon = []
    lon_segments = []
    for start, end in pre_semgentation:
        _,temp_poly_lat,temp_lat_segments = segment_series(time,lat,start,end)
        _,temp_polynomials_lon,temp_lon_segments = segment_series(time,lon,start,end)
        polynomials_lat = polynomials_lat + temp_poly_lat
        lat_segments = lat_segments + temp_lat_segments
        polynomials_lon = polynomials_lon + temp_polynomials_lon
        lon_segments = lon_segments + temp_lon_segments
    return polynomials_lat, lat_segments, polynomials_lon, lon_segments



def segment_trajectory_dyn_old(lon, lat, time,start_index, end_index = None, segment_upper_bound=None):
    _,polynomials_lat,lat_segments = segment_series(time,lat,start_index,len(time))
    _,polynomials_lon,lon_segments = segment_series(time,lon,start_index,len(time))
    print(lat_segments)
    return polynomials_lat, lat_segments, polynomials_lon, lon_segments

def euclidean_dist(x1,x2,y1,y2):
    return np.sqrt(pow(x2-x1,2)+ pow(y2-y1,2))

def find_jumps_(lon, lat):
     #Hyper parameters
    alpha = 30
    delta = 0.9
    
    dist_arr = []
    angle_arr = []
    candidate_jumps = []
    avg = 0
    #Calculate distance between each point and the angle between three points
    for i in range(0, len(lon)-1):
        dist_arr.append(euclidean_dist(lon[i+1],lon[i],lat[i+1],lat[i]))
        if i != 0: 
            dist = euclidean_dist(lon[i+1],lon[i-1],lat[i+1],lat[i-1])
            angle_arr.append((-pow(dist,2)+pow(dist_arr[i - 1],2)+pow(dist_arr[i],2))/(2*dist_arr[i - 1]*dist_arr[i])) #cosine relation
    maxv = max(dist_arr)
    for i in range(len(dist_arr)-1):
        avg = avg + (dist_arr[i]-avg)/(i+1)
    for i in range (1, len(lon)-1):
        #Distance threshold is dependent on the avg distance for all points
        if (angle_arr[i-1] >= 180 + alpha or angle_arr[i-1] <= 180 - alpha):
            if dist_arr[i - 1] >= delta*(maxv-avg) and dist_arr[i] >= delta*(maxv-avg):
                candidate_jumps.append((i,'o'))
            elif dist_arr[i - 1] >= delta*(maxv-avg) and dist_arr[i] < delta*(maxv-avg):
                candidate_jumps.append((i, 'l'))
            elif dist_arr[i - 1] < delta*(maxv-avg) and dist_arr[i] >= delta*(maxv-avg):
                candidate_jumps.append((i,'r'))     
    return construct_segments_(candidate_jumps, len(lon) - 1)

def construct_segments_(jumps,end):
    if jumps == []:
        return [(0,end)]
    segments = []
    start = 0  
    for i, type in jumps:
        if type == 'o':
            if start != i:
                segments.append((start, i-1))
            segments.append((i,i))
            start = i+1
        elif type == 'l':
            segments.append((start,i))
            start = i+1
        elif type == 'r':
            segments.append((start,i-1))
            start = i
    segments.append((jumps[-1]+1,end))
    return segments



def find_jumps(lon,lat):
    #Hyper parameters
    alpha = 30
    delta = 0.9

    dist_arr = []
    angle_arr = []
    candidate_jumps = []
    avg = 0
    #Calculate distance between each point and the between three points
    for i in range(0, len(lon)-1):
        dist_arr.append(euclidean_dist(lon[i+1],lon[i],lat[i+1],lat[i]))
        if i != 0: 
            dist = euclidean_dist(lon[i+1],lon[i-1],lat[i+1],lat[i-1])
            angle_arr.append((-pow(dist,2)+pow(dist_arr[i - 1],2)+pow(dist_arr[i],2))/(2*dist_arr[i - 1]*dist_arr[i])) #cosine relation
    maxv = max(dist_arr)
    for i in range(len(dist_arr)-1):
        avg = avg + (dist_arr[i]-avg)/(i+1)
    for i in range (1, len(lon)-1):
        #Distance threshold is dependent on the avg distance for all points
        if (angle_arr[i-1] >= 180 + alpha or angle_arr[i-1] <= 180 - alpha) and dist_arr[i-1] >= delta*(maxv-avg): 
            candidate_jumps.append(i)
    return construct_segments(candidate_jumps, len(lon) - 1)

def construct_segments(jumps,end):
    if jumps == []:
        return [(0,end)]
    segments = []
    start = 0
    for i in jumps:
        if start != i:
            segments.append((start, i-1))
        segments.append((i,i))
        start = i+1
    segments.append((jumps[-1]+1,end))
    return segments

def segment_series_simul(time, x, y, start, end):
    polynomial_x = fit_polynomial(time[start:end],x[start:end])
    polynomial_y = fit_polynomial(time[start:end],y[start:end])
    error_x = calc_error_SSE(start,end,x,polynomial_x)
    error_y = calc_error_SSE(start,end,y,polynomial_y)
    if(end-start < 2*N): 
        return error_x + error_y, [(polynomial_x,polynomial_y)],[(start,end)]
    else:
        q = (end + start)//2
        sum_error1, polynomial_list1, segments1 = segment_series(time, x, start,q-1)
        sum_error2, polynomial_list2, segments2 = segment_series(time, x, q,end)
        sum_error = sum_error2 + sum_error1
        if(sum_error < error_x+error_y):
            return sum_error, polynomial_list1 + polynomial_list2, segments1 + segments2
        else:
            return error_x + error_y [(polynomial_x,polynomial_y)], [(start, end)]


def segment_series(time, x, start, end):
    #print(f"{start}, {end}")
    if start == end:
        return 0, [np.polynomial.Polynomial(x[start])], [(start,end)]
    if end - start == 1:
        a = (x[end]-x[start])/(time[end]-time[start])
        return 0, [np.polynomial.Polynomial([x[start]-time[start]*a,a])], [(start,end)]
    polynomial = fit_polynomial(time[start:end],x[start:end])
    error = calc_error_SSE(start,end,x,polynomial)
    if(end-start < 2*N): 
        return error, [polynomial],[(start,end)]
    else:
        q = (end + start)//2
        sum_error1, polynomial_list1, segments1 = segment_series(time, x, start,q-1)
        sum_error2, polynomial_list2, segments2 = segment_series(time, x, q,end)
        sum_error = (sum_error2 + sum_error1)
        if(sum_error < error):
            return sum_error, polynomial_list1 + polynomial_list2, segments1 + segments2
        else:
            return error, [polynomial], [(start, end)]
 
E = np.exp(1)

def calc_error_SSE(start_index, end_index, values, polynomial):
    #Hyperparameters
    a = 25
    decay = 0.2
    #size_penalty = 10
    #parameter_penalty = 2
    
    result = 0
    for i in range(start_index,end_index): #calc SSE
        result += pow(values[i] - polynomial(i),2)
    return result + pow(a*E,-decay*(end_index - start_index - 1))

def calc_error_MSE(start_index, end_index, values, polynomial):
    #Hyperparameters
    a = 10
    decay = 0.4
    #size_penalty = 10
    #parameter_penalty = 2
    count = 1
    result = 0
    for i in range(start_index,end_index): #calc SSE
        result = result + (pow(values[i] - polynomial(i),2)-result)/count
        count += 1
    return result + pow(a*E,-decay*(end_index - start_index - 1))


def segment_trajectory(lon, lat, time, start_index=0, end_index=None, segment_upper_bound=None):

    segment_start = start_index
    segment_end = len(lon) if end_index is None else end_index
    segment_upper_bound = segment_end if segment_upper_bound is None else segment_upper_bound

    # These parameters definitely need to be examined further, do not just take them as the final values!!!
    lon_max_error = abs(max(lon[segment_start:segment_end]) - min(lon[segment_start:segment_end])) * 1.5
    # lat_max_error = abs(max(lat[segment_start:segment_end]) - min(lat[segment_start:segment_end])) * 1.5
    lat_max_error = 0.00001398 # this should be about 1 meter in latitudinal distance

    lon_polynomial = np.polynomial.Polynomial.fit(time[segment_start:segment_end], lon[segment_start:segment_end], deg=1, full=True)
    lat_polynomial = np.polynomial.Polynomial.fit(time[segment_start:segment_end], lat[segment_start:segment_end], deg=1, full=True)

    end_criterion = ((lon_polynomial[1][0][0]/(segment_end - segment_start) < lon_max_error) and (lat_polynomial[1][0][0]/(segment_end - segment_start) < lat_max_error)) or (segment_end - segment_start <= 10)

    
    if segment_end < segment_upper_bound:
        return segment_trajectory(lon, lat, time, segment_start, segment_end + max((segment_upper_bound - segment_end) // 2, 1), segment_upper_bound)
    elif end_criterion:
        lon_segments.append(lon[segment_start:segment_end])
        lat_segments.append(lat[segment_start:segment_end])
        time_segments.append(time[segment_start:segment_end])
        if segment_end < len(lon):
            return segment_trajectory(lon, lat, time, segment_end)
        return lon_segments, lat_segments, time_segments
    else :
        segment_end = segment_start + max((segment_end - segment_start) // 2, 1)
        segment_upper_bound = segment_end
        return segment_trajectory(lon, lat, time, segment_start, segment_end, segment_upper_bound)


create_visualisation()