import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
from typing import List, Tuple, Optional


class DataLoader:
    @staticmethod
    def load_financial_data(file_path: str) -> pd.DataFrame:
        """Load financial data from CSV file."""
        return pd.read_csv(file_path)


class Sorter:
    @staticmethod
    def merge_sort(arr: List[float]) -> None:
        """Implement merge sort algorithm."""
        if len(arr) > 1:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]

            Sorter.merge_sort(left)
            Sorter.merge_sort(right)

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


class TrendAnalyzer:
    @staticmethod
    def max_subarray(arr: np.ndarray) -> Tuple[float, int, int]:
        """Find maximum subarray sum and its indices."""
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


class AnomalyDetector:
    @staticmethod
    def date_to_numeric(date_str: str) -> int:
        """Convert date string to ordinal number."""
        return datetime.strptime(date_str, '%Y-%m-%d').toordinal()

    @staticmethod
    def euclidean_distance(p1: Tuple[int, float], p2: Tuple[int, float]) -> float:
        """Calculate scaled Euclidean distance between two points."""
        date_scale = 1 / 365
        return math.sqrt(((p1[0] - p2[0]) * date_scale) ** 2 + ((p1[1] - p2[1])) ** 2)

    def closest_pair(self, points: List[Tuple[int, float]], min_date_diff: int = 1) -> float:
        """Find the closest pair of points considering minimum date difference."""

        def closest_pair_rec(pts):
            if len(pts) <= 3:
                distances = []
                pts_array = np.array(pts)
                for i in range(len(pts)):
                    current = pts_array[i]
                    others = pts_array[i + 1:]
                    time_diffs = np.abs(others[:, 0] - current[0])
                    valid_indices = time_diffs >= min_date_diff

                    for point in others[valid_indices]:
                        distances.append(self.euclidean_distance(current, point))

                return min(distances) if distances else float('inf')

            mid = len(pts) // 2
            mid_x = pts[mid][0]

            d_left = closest_pair_rec(pts[:mid])
            d_right = closest_pair_rec(pts[mid:])
            d = min(d_left, d_right)

            strip = [p for p in pts if abs(p[0] - mid_x) < d]
            strip.sort(key=lambda p: p[1])

            min_d_strip = float('inf')
            strip_array = np.array(strip)

            for i in range(len(strip)):
                current = strip_array[i]
                end_idx = min(i + 8, len(strip))
                if i < end_idx:
                    next_points = strip_array[i + 1:end_idx]
                    time_diffs = np.abs(next_points[:, 0] - current[0])
                    valid_indices = time_diffs >= min_date_diff

                    for point in next_points[valid_indices]:
                        dist = self.euclidean_distance(current, point)
                        min_d_strip = min(min_d_strip, dist)

            return min(d, min_d_strip)

        if len(points) < 2:
            return float('inf')

        points = list(set(points))
        points.sort()
        return closest_pair_rec(points)

    def find_anomalies(self, df: pd.DataFrame, threshold: Optional[float] = None) -> Tuple[List[Tuple], float]:
        """Find anomalies in the financial data."""
        numeric_dates = df['date'].map(self.date_to_numeric).values
        close_prices = df['close'].values
        numeric_points = list(zip(numeric_dates, close_prices))
        min_distance = self.closest_pair(numeric_points, min_date_diff=1)

        if threshold is None:
            threshold = min_distance * 1.5

        anomalies = []
        points_array = np.array(numeric_points)

        for i in range(len(points_array)):
            end_idx = min(i + 8, len(points_array))
            if i < end_idx:
                current_point = points_array[i]
                next_points = points_array[i + 1:end_idx]
                time_diffs = np.abs(next_points[:, 0] - current_point[0])
                valid_indices = time_diffs >= 1

                if np.any(valid_indices):
                    valid_points = next_points[valid_indices]
                    for point in valid_points:
                        dist = self.euclidean_distance(current_point, point)
                        if dist < threshold:
                            anomalies.append((tuple(current_point), tuple(point), dist))

        return anomalies, min_distance


class Visualizer:
    @staticmethod
    def generate_report(df: pd.DataFrame, start_idx: int, end_idx: int, title: str = "Stock Analysis Report") -> None:
        """Generate and display a visual report."""
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['close'], label="Stock Price")
        plt.axvspan(df['date'][start_idx], df['date'][end_idx], color='red', alpha=0.3, label="Max Gain Period")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(title)
        plt.legend()
        plt.show()


class StockAnalyzer:
    def __init__(self):
        self.data_loader = DataLoader()
        self.sorter = Sorter()
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = Visualizer()

    def analyze_stock_data(self, file_path: str) -> None:
        """Perform complete stock analysis."""
        # Load data
        df = self.data_loader.load_financial_data(file_path)

        # Sort prices
        prices = list(df['close'])
        self.sorter.merge_sort(prices)

        # Analyze trends
        price_changes = np.diff(df['close'])
        max_gain, start_idx, end_idx = self.trend_analyzer.max_subarray(price_changes)
        print(f"Maximum gain: {max_gain}, from {df['date'][start_idx]} to {df['date'][end_idx]}")

        # Detect anomalies
        anomalies, min_distance = self.anomaly_detector.find_anomalies(df)
        print(f"\nMinimum distance found: {min_distance}")
        print("Potential anomalies:")
        for p1, p2, dist in anomalies:
            date1 = datetime.fromordinal(int(p1[0])).strftime('%Y-%m-%d')
            date2 = datetime.fromordinal(int(p2[0])).strftime('%Y-%m-%d')
            print(f"Points: ({date1}, {p1[1]:.2f}) and ({date2}, {p2[1]:.2f}) - Distance: {dist:.2f}")

        # Visualize results
        self.visualizer.generate_report(df, start_idx, end_idx,
                                        title="Period of Maximum Gain in Stock Prices")


if __name__ == "__main__":
    analyzer = StockAnalyzer()
    analyzer.analyze_stock_data('SP 500 Stock Prices 2014-2017.csv')

    # Sort Toy example
    print('\n Toy examples')
    toy_data = [5, 3, 8, 1, 2]
    Sorter.merge_sort(toy_data)
    print(f"Sorted data: {toy_data}")

    # Max gain toy example
    toy_changes = np.array([3, -2, 5, -1, 6])
    max_gain, start_idx, end_idx = TrendAnalyzer.max_subarray(toy_changes)
    print(f"Max Gain: {max_gain}, from index {start_idx} to {end_idx}")

    # Find anomalies toy example
    sample_df = pd.DataFrame({
        'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'close': [100, 200, 105]
    })
    anomaly_detector = AnomalyDetector()
    anomalies, min_distance = anomaly_detector.find_anomalies(sample_df)
    print(f"Anomalies: {anomalies}")

    # Generate Report Toy Example
    toy_df = pd.DataFrame({
        'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
        'close': [100, 200, 105]
    })

    price_changes = np.diff(toy_df['close'])
    max_gain, start_idx, end_idx = TrendAnalyzer.max_subarray(price_changes)
    Visualizer.generate_report(sample_df, start_idx, end_idx, title="Toy Example: Stock Price Report")