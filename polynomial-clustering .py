
import numpy as np
from sklearn.cluster import DBSCAN



def evaluate_polynomial(coefficients, time):
    return np.polynomial.polynomial.polyval(time, coefficients)


def get_time_interval(polynomials):

    times = polynomials[2]

    start_time = min(segment[0] for segment in times)
    end_time = max(segment[1] for segment in times)

    return start_time, end_time



def evaluate_trajectory(polynomials, time):

    lons, lats, times = polynomials

    segments = sorted(
        zip(lons, lats, times),
        key=lambda segment: segment[2][0]
    )

    for i, (lon_coeff, lat_coeff, interval) in enumerate(segments):

        start_time, end_time = interval
        is_last_segment = i == len(segments) - 1

        if start_time <= time < end_time or (
            is_last_segment and start_time <= time <= end_time
        ):

            lon = evaluate_polynomial(lon_coeff, time)
            lat = evaluate_polynomial(lat_coeff, time)

            return lon, lat

    return None



def haversine_distance(lon1, lat1, lon2, lat2):

    earth_radius = 6371000

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2]
    )

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    return earth_radius * c



def create_time_samples(start_time, end_time, step_length):
  

    if step_length <= 0:
        raise ValueError("step_length must be positive")

    if end_time < start_time:
        return np.array([])

    if end_time == start_time:
        return np.array([start_time])

    sample_times = np.arange(
        start_time,
        end_time,
        step_length
    )

    if not np.isclose(sample_times[-1], end_time):
        sample_times = np.append(sample_times, end_time)

    return sample_times


# Returns infinity if the trajectories do not overlap sufficiently in time
def trajectory_distance(
    trajectory1,
    trajectory2,
    step_length=2,
    min_overlap_ratio=0.5
):

    start1, end1 = get_time_interval(trajectory1)
    start2, end2 = get_time_interval(trajectory2)

    # Find the overlapping time interval
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)

    if overlap_end <= overlap_start:
        return np.inf

    duration1 = end1 - start1
    duration2 = end2 - start2

    shorter_duration = min(duration1, duration2)

    if shorter_duration <= 0:
        return np.inf

    overlap_duration = overlap_end - overlap_start

    overlap_ratio = overlap_duration / shorter_duration

    # Requires sufficient temporal overlap
    if overlap_ratio < min_overlap_ratio:
        return np.inf

    sample_times = create_time_samples(
        overlap_start,
        overlap_end,
        step_length
    )

    distances = []

    for time in sample_times:

        position1 = evaluate_trajectory(
            trajectory1,
            time
        )

        position2 = evaluate_trajectory(
            trajectory2,
            time
        )

        if position1 is None or position2 is None:
            continue

        lon1, lat1 = position1
        lon2, lat2 = position2

        distance = haversine_distance(
            lon1,
            lat1,
            lon2,
            lat2
        )

        distances.append(distance)

    if not distances:
        return np.inf

    return float(np.mean(distances))



def clustering_query(
    data,
    eps=100,
    min_samples=3,
    step_length=2,
    min_overlap_ratio=0.5
):
    """
    Cluster polynomial trajectories using DBSCAN.

    Parameters:
        data:
            Dictionary containing polynomial trajectories:
            {
                filename: [lons, lats, times],
                ...
            }

        eps:
            Maximum average distance in meters for two
            trajectories to be considered neighbors.

        min_samples:
            Minimum number of trajectories needed to
            form a cluster.

        step_length:
            Time interval between sampled positions.

        min_overlap_ratio:
            Minimum fraction of the shorter trajectory's
            duration that must overlap.

    Returns:
        labels:
            Dictionary assigning a cluster ID to each
            trajectory.

        clusters:
            Dictionary grouping filenames by cluster ID.

        Cluster ID -1 represents noise.
    """

    filenames = list(data.keys())

    number_of_trajectories = len(filenames)

    if number_of_trajectories == 0:
        return {}, {}

   
    distance_matrix = np.zeros(
        (number_of_trajectories, number_of_trajectories)
    )

    for i in range(number_of_trajectories):

        for j in range(i + 1, number_of_trajectories):

            trajectory1 = data[filenames[i]]
            trajectory2 = data[filenames[j]]

            distance = trajectory_distance(
                trajectory1,
                trajectory2,
                step_length=step_length,
                min_overlap_ratio=min_overlap_ratio
            )

            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

   
    no_overlap_distance = max(eps * 1000, 1e9)

    distance_matrix[~np.isfinite(distance_matrix)] = (
        no_overlap_distance
    )

   
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed"
    )

    cluster_labels = dbscan.fit_predict(distance_matrix)

    
    labels = {}

    for filename, cluster_id in zip(filenames, cluster_labels):
        labels[filename] = int(cluster_id)

    clusters = {}

    for filename, cluster_id in labels.items():

        if cluster_id not in clusters:
            clusters[cluster_id] = []

        clusters[cluster_id].append(filename)

    return labels, clusters


