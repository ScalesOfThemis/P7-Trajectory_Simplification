
import numpy as np



def evaluate_polynomial(coefficients, time):
    
    return np.polynomial.polynomial.polyval(time, coefficients)


# polynomials = [lons, lats, times]
def get_time_interval(polynomials):
    times = polynomials[2]

    start_time = min(segment[0] for segment in times)
    end_time = max(segment[1] for segment in times)

    return start_time, end_time



def evaluate_trajectory(polynomials, time):

    lons, lats, times = polynomials

    for i, (lon_coeff, lat_coeff, interval) in enumerate(
        zip(lons, lats, times)
    ):

        start_time, end_time = interval

        is_last_segment = i == len(times) - 1

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



def similarity_query(
    data,
    target,
    time_range=None,
    delta=100,
    step_length=2,
    similarity_threshold=0.8
):
    """
    Find trajectories similar to a target polynomial trajectory.

    Parameters:
        data:
            Dictionary containing polynomial trajectories:
            {
                filename: [lons, lats, times],
                ...
            }

        target:
            Target polynomial trajectory.

        time_range:
            Optional (start_time, end_time).

        delta:
            Maximum allowed distance in meters.

        step_length:
            Time interval between sampled positions.

        similarity_threshold:
            Minimum fraction of positions that must be
            within delta meters.

    Returns:
        List containing:
        (filename, trajectory, similarity_score)
    """

    if time_range is None:
        query_start, query_end = get_time_interval(target)
    else:
        query_start, query_end = time_range

    sample_times = create_time_samples(
        query_start,
        query_end,
        step_length
    )

    results = []

    for filename, candidate in data.items():

        candidate_start, candidate_end = get_time_interval(
            candidate
        )

        if (
            candidate_start > query_start
            or candidate_end < query_end
        ):
            continue

        distances = []

        for time in sample_times:

            target_position = evaluate_trajectory(
                target,
                time
            )

            candidate_position = evaluate_trajectory(
                candidate,
                time
            )

            if (
                target_position is None
                or candidate_position is None
            ):
                continue

            target_lon, target_lat = target_position
            candidate_lon, candidate_lat = candidate_position

            distance = haversine_distance(
                target_lon,
                target_lat,
                candidate_lon,
                candidate_lat
            )

            distances.append(distance)

        if not distances:
            continue

        # Calculate fraction of positions within delta.
        similar_positions = sum(
            distance <= delta
            for distance in distances
        )

        similarity = similar_positions / len(distances)

        if similarity >= similarity_threshold:

            results.append(
                (filename, candidate, similarity)
            )

    return results



# Compare query results 

def compare_results(original_results, simplified_results):

    original = {
        result[0] if isinstance(result, tuple) else result
        for result in original_results
    }

    simplified = {
        result[0] if isinstance(result, tuple) else result
        for result in simplified_results
    }

    if not original and not simplified:
        return 1.0, 1.0, 1.0

    true_positives = len(original & simplified)

    precision = (
        true_positives / len(simplified)
        if simplified else 0
    )

    recall = (
        true_positives / len(original)
        if original else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return precision, recall, f1

