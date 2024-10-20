import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

def load_financial_data(file_path):
    data = pd.read_csv(file_path)
    return data


def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]

        merge_sort(left)
        merge_sort(right)

        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1


def max_subarray(arr):
    max_ending_here = max_so_far = arr[0]
    start = end = s = 0

    for i in range(1, len(arr)):
        if arr[i] > max_ending_here + arr[i]:
            max_ending_here = arr[i]
            s = i
        else:
            max_ending_here += arr[i]

        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            start = s
            end = i

    return max_so_far, start, end


def date_to_numeric(date_str):
    """Convert date string to ordinal number."""
    return datetime.strptime(date_str, '%Y-%m-%d').toordinal()


def euclidean_distance(p1, p2):
    # Scale the distance calculation to handle the different magnitudes
    # between dates and prices
    date_scale = 1 / 365  # Scale dates to roughly handle year differences
    return math.sqrt(((p1[0] - p2[0]) * date_scale) ** 2 + ((p1[1] - p2[1])) ** 2)


def closest_pair(points, min_date_diff=1):
    def closest_pair_rec(pts):
        if len(pts) <= 3:
            distances = []
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    # Only calculate distance if points are at least min_date_diff apart
                    if abs(pts[i][0] - pts[j][0]) >= min_date_diff:
                        distances.append(euclidean_distance(pts[i], pts[j]))
            return min(distances) if distances else float('inf')

        mid = len(pts) // 2
        mid_x = pts[mid][0]

        # Recursive calls for left and right halves
        d_left = closest_pair_rec(pts[:mid])
        d_right = closest_pair_rec(pts[mid:])

        d = min(d_left, d_right)

        # Create the strip and sort it by y-coordinate (price)
        strip = [p for p in pts if abs(p[0] - mid_x) < d]
        strip.sort(key=lambda p: p[1])

        min_d_strip = float('inf')
        for i in range(len(strip)):
            # Only look at a limited window of points ahead
            j = i + 1
            while j < len(strip) and j < i + 8:  # Limit to 7 comparisons
                if abs(strip[i][0] - strip[j][0]) >= min_date_diff:
                    dist = euclidean_distance(strip[i], strip[j])
                    min_d_strip = min(min_d_strip, dist)
                j += 1

        return min(d, min_d_strip)

    if not points:
        return float('inf')

    # Remove duplicates and sort by date
    points = list(set(points))
    points.sort()  # Sort by date (x-coordinate)

    # If we have very few points after removing duplicates
    if len(points) < 2:
        return float('inf')

    return closest_pair_rec(points)


def find_anomalies(df, threshold=None):
    # Vectorized conversion of dates to numeric values
    numeric_dates = df['date'].map(date_to_numeric).values
    close_prices = df['close'].values

    # Create points list using numpy operations
    numeric_points = list(zip(numeric_dates, close_prices))

    # Sort points by date
    numeric_points.sort()

    # Find the closest pair distance
    min_distance = closest_pair(numeric_points, min_date_diff=1)

    if threshold is None:
        threshold = min_distance * 1.5

    # Find anomalies more efficiently
    anomalies = []
    points_array = np.array(numeric_points)

    # Use numpy broadcasting for faster distance calculations
    for i in range(len(points_array)):
        # Only look at a window of next 7 points to maintain efficiency
        end_idx = min(i + 8, len(points_array))
        if i < end_idx:
            current_point = points_array[i]
            next_points = points_array[i + 1:end_idx]

            # Calculate time differences
            time_diffs = np.abs(next_points[:, 0] - current_point[0])
            valid_indices = time_diffs >= 1

            if np.any(valid_indices):
                valid_points = next_points[valid_indices]
                for point in valid_points:
                    dist = euclidean_distance(current_point, point)
                    if dist < threshold:
                        anomalies.append((tuple(current_point), tuple(point), dist))

    return anomalies, min_distance


def generate_report(df, start_idx, end_idx, title="Stock Analysis Report"):
    plt.figure(figsize=(10,6))
    plt.plot(df['date'], df['close'], label="Stock Price")
    plt.axvspan(df['date'][start_idx], df['date'][end_idx], color='red', alpha=0.3, label="Max Gain Period")
    plt.xlabel('date')
    plt.ylabel('price')
    plt.title(title)
    plt.legend()
    plt.show()


df = load_financial_data('SP 500 Stock Prices 2014-2017.csv')
prices = list(df['close'])
merge_sort(prices)
# print(prices)

price_changes = np.diff(df['close'])  # Daily price change
max_gain, start_idx, end_idx = max_subarray(price_changes)
print(f"Maximum gain: {max_gain}, from {df['date'][start_idx]} to {df['date'][end_idx]}")

anomalies, min_distance = find_anomalies(df)
print(f"\nMinimum distance found: {min_distance}")
print("Potential anomalies:")
for p1, p2, dist in anomalies:
    date1 = datetime.fromordinal(int(p1[0])).strftime('%Y-%m-%d')
    date2 = datetime.fromordinal(int(p2[0])).strftime('%Y-%m-%d')
    print(f"Points: ({date1}, {p1[1]:.2f}) and ({date2}, {p2[1]:.2f}) - Distance: {dist:.2f}")

# generate_report(df, start_idx, end_idx, title="Period of Maximum Gain in Stock Prices")