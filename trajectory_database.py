"""
    python3 trajectory_database.py --import-geolife "path/to/file.plt" --trajectory-id "traj_001"
"""

import argparse
import csv
import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS trajectories (
    trajectory_id TEXT PRIMARY KEY,
    source_dataset TEXT,
    source_file TEXT,
    user_id TEXT,
    coordinate_system TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS original_points (
    trajectory_id TEXT NOT NULL,
    point_index INTEGER NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    altitude REAL,
    timestamp TEXT,
    PRIMARY KEY (trajectory_id, point_index),
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS simplifications (
    simplification_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trajectory_id TEXT NOT NULL,
    method TEXT NOT NULL,
    max_degree INTEGER,
    error_tolerance REAL,
    coordinate_system TEXT,
    parameterization TEXT,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trajectory_id) REFERENCES trajectories(trajectory_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    simplification_id INTEGER NOT NULL,
    segment_index INTEGER NOT NULL,
    start_point_index INTEGER,
    end_point_index INTEGER,
    parameter_start REAL NOT NULL DEFAULT 0.0,
    parameter_end REAL NOT NULL DEFAULT 1.0,
    polynomial_degree INTEGER NOT NULL,
    FOREIGN KEY (simplification_id) REFERENCES simplifications(simplification_id)
        ON DELETE CASCADE,
    UNIQUE (simplification_id, segment_index),
    CHECK (polynomial_degree >= 0),
    CHECK (parameter_end >= parameter_start)
);

CREATE TABLE IF NOT EXISTS polynomial_coefficients (
    segment_id INTEGER NOT NULL,
    coordinate TEXT NOT NULL,
    coefficient_order INTEGER NOT NULL,
    coefficient_value REAL NOT NULL,
    PRIMARY KEY (segment_id, coordinate, coefficient_order),
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
        ON DELETE CASCADE,
    CHECK (coordinate IN ('x', 'y')),
    CHECK (coefficient_order >= 0)
);

CREATE INDEX IF NOT EXISTS idx_original_points_order
    ON original_points(trajectory_id, point_index);
CREATE INDEX IF NOT EXISTS idx_simplifications_trajectory
    ON simplifications(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_segments_order
    ON segments(simplification_id, segment_index);
"""


class TrajectoryDB:
    def __init__(self, db_path="trajectory_database.db"):
        self.db_path = str(db_path)
        with self.connect() as con:
            con.executescript(SCHEMA)

    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def add_trajectory(self, trajectory_id, source_dataset="unknown",
                       source_file=None, user_id=None,
                       coordinate_system="latitude_longitude"):
        with self.connect() as con:
            con.execute("""
                INSERT INTO trajectories
                (trajectory_id, source_dataset, source_file, user_id, coordinate_system)
                VALUES (?, ?, ?, ?, ?)
            """, (str(trajectory_id), source_dataset, source_file,
                  user_id, coordinate_system))

    def add_original_points(self, trajectory_id, points):
        """
        points can be dictionaries:
          {"latitude": 55.6, "longitude": 12.5, "timestamp": "...", "altitude": 10}
        or tuples:
          (latitude, longitude)
          (latitude, longitude, timestamp)
          (latitude, longitude, altitude, timestamp)
        """
        rows = []
        for index, point in enumerate(points):
            if isinstance(point, dict):
                lat = point["latitude"]
                lon = point["longitude"]
                altitude = point.get("altitude")
                timestamp = point.get("timestamp")
            else:
                values = tuple(point)
                if len(values) == 2:
                    lat, lon = values
                    altitude, timestamp = None, None
                elif len(values) == 3:
                    lat, lon, timestamp = values
                    altitude = None
                elif len(values) == 4:
                    lat, lon, altitude, timestamp = values
                else:
                    raise ValueError("Point tuple must have 2, 3, or 4 values.")

            rows.append((str(trajectory_id), index, float(lat), float(lon),
                         None if altitude is None else float(altitude), timestamp))

        with self.connect() as con:
            con.executemany("""
                INSERT INTO original_points
                (trajectory_id, point_index, latitude, longitude, altitude, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, rows)
        return len(rows)

    def add_simplification(self, trajectory_id, segments,
                           method="parametric_polynomial", max_degree=None,
                           error_tolerance=None, coordinate_system=None,
                           parameterization="normalized_0_to_1_per_segment",
                           notes=None):
        """
        Each segment must provide:
          x_coefficients: [a0, a1, ...]
          y_coefficients: [b0, b1, ...]
        Coefficients are ordered from constant term upwards:
          [a0, a1, a2] means a0 + a1*t + a2*t**2.
        """
        if not segments:
            raise ValueError("At least one segment is required.")

        with self.connect() as con:
            cur = con.execute("""
                INSERT INTO simplifications
                (trajectory_id, method, max_degree, error_tolerance,
                 coordinate_system, parameterization, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (str(trajectory_id), method, max_degree, error_tolerance,
                  coordinate_system, parameterization, notes))
            simplification_id = cur.lastrowid

            for fallback_index, segment in enumerate(segments):
                xs = [float(v) for v in segment["x_coefficients"]]
                ys = [float(v) for v in segment["y_coefficients"]]
                if not xs or not ys:
                    raise ValueError("Each segment needs x and y coefficients.")

                degree = int(segment.get("degree", max(len(xs), len(ys)) - 1))
                if max(len(xs), len(ys)) - 1 > degree:
                    raise ValueError("Coefficient list exceeds declared degree.")

                cur = con.execute("""
                    INSERT INTO segments
                    (simplification_id, segment_index, start_point_index,
                     end_point_index, parameter_start, parameter_end, polynomial_degree)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    simplification_id,
                    int(segment.get("segment_index", fallback_index)),
                    segment.get("start_point_index"),
                    segment.get("end_point_index"),
                    float(segment.get("parameter_start", 0.0)),
                    float(segment.get("parameter_end", 1.0)),
                    degree,
                ))
                segment_id = cur.lastrowid

                coefficient_rows = (
                    [(segment_id, "x", order, value) for order, value in enumerate(xs)]
                    + [(segment_id, "y", order, value) for order, value in enumerate(ys)]
                )
                con.executemany("""
                    INSERT INTO polynomial_coefficients
                    (segment_id, coordinate, coefficient_order, coefficient_value)
                    VALUES (?, ?, ?, ?)
                """, coefficient_rows)

        return simplification_id

    def get_original_points(self, trajectory_id):
        with self.connect() as con:
            rows = con.execute("""
                SELECT point_index, latitude, longitude, altitude, timestamp
                FROM original_points
                WHERE trajectory_id = ?
                ORDER BY point_index
            """, (str(trajectory_id),)).fetchall()
        return [dict(row) for row in rows]

    def list_simplifications(self, trajectory_id):
        with self.connect() as con:
            rows = con.execute("""
                SELECT * FROM simplifications
                WHERE trajectory_id = ?
                ORDER BY simplification_id
            """, (str(trajectory_id),)).fetchall()
        return [dict(row) for row in rows]

    def get_simplification(self, simplification_id):
        with self.connect() as con:
            metadata = con.execute("""
                SELECT * FROM simplifications WHERE simplification_id = ?
            """, (int(simplification_id),)).fetchone()
            if metadata is None:
                raise KeyError("Simplification not found.")

            segment_rows = con.execute("""
                SELECT * FROM segments
                WHERE simplification_id = ?
                ORDER BY segment_index
            """, (int(simplification_id),)).fetchall()

            segments = []
            for segment in segment_rows:
                coeffs = con.execute("""
                    SELECT coordinate, coefficient_order, coefficient_value
                    FROM polynomial_coefficients
                    WHERE segment_id = ?
                    ORDER BY coordinate, coefficient_order
                """, (segment["segment_id"],)).fetchall()

                x_coefficients = [
                    row["coefficient_value"] for row in coeffs
                    if row["coordinate"] == "x"
                ]
                y_coefficients = [
                    row["coefficient_value"] for row in coeffs
                    if row["coordinate"] == "y"
                ]

                item = dict(segment)
                item["x_coefficients"] = x_coefficients
                item["y_coefficients"] = y_coefficients
                segments.append(item)

        return {"metadata": dict(metadata), "segments": segments}

    def import_geolife_plt(self, plt_path, trajectory_id, user_id=None,
                           skip_header_lines=6):
       
        plt_path = Path(plt_path)
        self.add_trajectory(
            trajectory_id=trajectory_id,
            source_dataset="GeoLife",
            source_file=str(plt_path),
            user_id=user_id,
            coordinate_system="latitude_longitude",
        )

        points = []
        with plt_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            for _ in range(skip_header_lines):
                next(f, None)

            for row in csv.reader(f):
                if len(row) < 7:
                    continue
                try:
                    latitude = float(row[0])
                    longitude = float(row[1])
                    altitude = float(row[3]) if row[3].strip() else None
                    timestamp = f"{row[5].strip()} {row[6].strip()}".strip()
                except (ValueError, IndexError):
                    continue

                points.append({
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude": altitude,
                    "timestamp": timestamp,
                })

        return self.add_original_points(trajectory_id, points)


def demo():
    """Use invented example data to test that storing and retrieval work."""
    db = TrajectoryDB()

    trajectory_id = "demo_trajectory_001"

    try:
        db.add_trajectory(
            trajectory_id, source_dataset="demo", user_id="demo_user"
        )
    except sqlite3.IntegrityError:
        print("Demo trajectory already exists. Delete the database to rerun from scratch.")
        return

    db.add_original_points(trajectory_id, [
        {"latitude": 55.6761, "longitude": 12.5683, "timestamp": "2026-01-01 10:00:00"},
        {"latitude": 55.6762, "longitude": 12.5685, "timestamp": "2026-01-01 10:01:00"},
        {"latitude": 55.6764, "longitude": 12.5688, "timestamp": "2026-01-01 10:02:00"},
    ])

    # Illustrative coefficients only; another program must calculate real ones.
    example_segments = [{
        "segment_index": 0,
        "start_point_index": 0,
        "end_point_index": 2,
        "degree": 2,
        "parameter_start": 0.0,
        "parameter_end": 1.0,
        "x_coefficients": [10.0, 3.0, 2.0],
        "y_coefficients": [20.0, 5.0, 4.0],
    }]

    simplification_id = db.add_simplification(
        trajectory_id,
        example_segments,
        max_degree=2,
        error_tolerance=5.0,
        coordinate_system="local_projected_metres",
        notes="Demo values; not fitted from the GPS points.",
    )

    print("\\nOriginal points:")
    for point in db.get_original_points(trajectory_id):
        print(point)

    print("\\nStored simplification:")
    print(db.get_simplification(simplification_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-geolife", help="Path to one GeoLife .plt file")
    parser.add_argument("--trajectory-id", help="ID for the imported trajectory")
    args = parser.parse_args()

    if args.import_geolife:
        if not args.trajectory_id:
            parser.error("--trajectory-id is required with --import-geolife")
        db = TrajectoryDB()
        number = db.import_geolife_plt(args.import_geolife, args.trajectory_id)
        print(f"Imported {number} original GPS points.")
    else:
        demo()
