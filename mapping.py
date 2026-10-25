"""Non-probabilistic 2-D occupancy mapping for CS 437 Lab 1 Step 6.

World coordinates use centimeters. At the initial pose, +x is straight ahead,
+y is to the car's left, and positive yaw is counter-clockwise.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


UNKNOWN = -1
FREE = 0
OCCUPIED = 1


@dataclass(frozen=True)
class Pose:
    x_cm: float = 0.0
    y_cm: float = 0.0
    yaw_rad: float = 0.0

    def integrate(self, velocity_cm_s: float, yaw_rate_rad_s: float, dt_s: float) -> "Pose":
        """Emulate localization by integrating measured/estimated velocity."""
        if dt_s < 0:
            raise ValueError("dt_s cannot be negative")
        mid_yaw = self.yaw_rad + yaw_rate_rad_s * dt_s / 2.0
        distance = velocity_cm_s * dt_s
        yaw = _wrap_angle(self.yaw_rad + yaw_rate_rad_s * dt_s)
        return Pose(
            self.x_cm + distance * math.cos(mid_yaw),
            self.y_cm + distance * math.sin(mid_yaw),
            yaw,
        )

    def advance_bicycle(
        self, distance_cm: float, steering_rad: float, wheelbase_cm: float = 12.0
    ) -> "Pose":
        """Advance an Ackermann-steered car using the bicycle motion model."""
        curvature = math.tan(steering_rad) / wheelbase_cm
        if abs(curvature) < 1e-9:
            return Pose(
                self.x_cm + distance_cm * math.cos(self.yaw_rad),
                self.y_cm + distance_cm * math.sin(self.yaw_rad),
                self.yaw_rad,
            )
        yaw_change = distance_cm * curvature
        return Pose(
            self.x_cm
            + (math.sin(self.yaw_rad + yaw_change) - math.sin(self.yaw_rad)) / curvature,
            self.y_cm
            + (math.cos(self.yaw_rad) - math.cos(self.yaw_rad + yaw_change)) / curvature,
            _wrap_angle(self.yaw_rad + yaw_change),
        )


@dataclass(frozen=True)
class RangeReading:
    angle_deg: float
    distance_cm: float | None


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class OccupancyGrid:
    """A square grid storing UNKNOWN, FREE, and OCCUPIED cells."""

    def __init__(self, size_cm: int = 400, resolution_cm: int = 2):
        if size_cm <= 0 or resolution_cm <= 0 or size_cm % resolution_cm:
            raise ValueError("size_cm must be positive and divisible by resolution_cm")
        self.resolution_cm = resolution_cm
        self.width = size_cm // resolution_cm
        self.origin_cm = -size_cm / 2.0
        self.cells = np.full((self.width, self.width), UNKNOWN, dtype=np.int8)

    def world_to_cell(self, x_cm: float, y_cm: float) -> tuple[int, int]:
        return (
            math.floor((x_cm - self.origin_cm) / self.resolution_cm),
            math.floor((y_cm - self.origin_cm) / self.resolution_cm),
        )

    def inside(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.width

    def _line_cells(
        self, x0_cm: float, y0_cm: float, x1_cm: float, y1_cm: float
    ) -> list[tuple[int, int]]:
        start = self.world_to_cell(x0_cm, y0_cm)
        end = self.world_to_cell(x1_cm, y1_cm)
        count = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
        cells = [
            (
                round(start[0] + (end[0] - start[0]) * index / count),
                round(start[1] + (end[1] - start[1]) * index / count),
            )
            for index in range(count + 1)
        ]
        return list(dict.fromkeys(cells))

    def update_scan(
        self,
        pose: Pose,
        readings: Iterable[RangeReading],
        *,
        max_range_cm: float = 150.0,
        sensor_offset_cm: float = 8.0,
        interpolate_max_gap_cm: float = 25.0,
        interpolate_max_delta_cm: float = 15.0,
    ) -> None:
        """Insert one ultrasonic sweep into the occupancy grid.

        A valid return shorter than max_range marks its endpoint occupied. Cells
        before that endpoint are free. Invalid/time-out readings are ignored so
        a sensor failure is never interpreted as clear space.
        """
        readings = list(readings)
        sensor_x = pose.x_cm + sensor_offset_cm * math.cos(pose.yaw_rad)
        sensor_y = pose.y_cm + sensor_offset_cm * math.sin(pose.yaw_rad)
        endpoints: list[tuple[RangeReading, float, float] | None] = []

        for reading in readings:
            distance = reading.distance_cm
            if distance is None or not math.isfinite(distance) or distance <= 0:
                endpoints.append(None)
                continue

            hit = distance < max_range_cm
            ray_distance = min(distance, max_range_cm)
            angle = pose.yaw_rad + math.radians(reading.angle_deg)
            end_x = sensor_x + ray_distance * math.cos(angle)
            end_y = sensor_y + ray_distance * math.sin(angle)
            ray = self._line_cells(sensor_x, sensor_y, end_x, end_y)
            free_cells = ray[:-1] if hit else ray
            for cell in free_cells:
                if self.inside(cell):
                    self.cells[cell[1], cell[0]] = FREE
            endpoints.append((reading, end_x, end_y) if hit else None)

        # Add occupied endpoints after all free rays so an adjacent beam cannot
        # erase an obstacle observed during the same sweep.
        for endpoint in endpoints:
            if endpoint is None:
                continue
            cell = self.world_to_cell(endpoint[1], endpoint[2])
            if self.inside(cell):
                self.cells[cell[1], cell[0]] = OCCUPIED

        # The manual asks us to interpolate between nearby returns. Only join
        # similar adjacent readings to avoid drawing walls across doorways.
        for first, second in zip(endpoints, endpoints[1:]):
            if first is None or second is None:
                continue
            first_reading, x1, y1 = first
            second_reading, x2, y2 = second
            if (
                abs(first_reading.distance_cm - second_reading.distance_cm)
                <= interpolate_max_delta_cm
                and math.hypot(x2 - x1, y2 - y1) <= interpolate_max_gap_cm
            ):
                for cell in self._line_cells(x1, y1, x2, y2):
                    if self.inside(cell):
                        self.cells[cell[1], cell[0]] = OCCUPIED

    def inflated_obstacles(self, clearance_cm: float) -> np.ndarray:
        """Return occupied cells expanded by the car's safety clearance."""
        blocked = np.zeros_like(self.cells, dtype=bool)
        radius = math.ceil(clearance_cm / self.resolution_cm)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if math.hypot(dx, dy) * self.resolution_cm > clearance_cm:
                    continue
                source_y0, source_y1 = max(0, -dy), min(self.width, self.width - dy)
                source_x0, source_x1 = max(0, -dx), min(self.width, self.width - dx)
                blocked[
                    source_y0 + dy : source_y1 + dy,
                    source_x0 + dx : source_x1 + dx,
                ] |= self.cells[source_y0:source_y1, source_x0:source_x1] == OCCUPIED
        blocked[:radius, :] = True
        blocked[-radius:, :] = True
        blocked[:, :radius] = True
        blocked[:, -radius:] = True
        return blocked

    def save(
        self,
        output_dir: str | Path,
        *,
        pose: Pose,
        readings: Iterable[RangeReading],
        route: Iterable[Pose] = (),
        goal: tuple[float, float] | None = None,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        readings = list(readings)
        route = list(route)

        # Preserve unknown/free/occupied state for later routing.
        np.save(output_dir / "occupancy.npy", self.cells)
        # The rubric explicitly accepts a binary 0/1 obstacle map.
        np.savetxt(
            output_dir / "obstacles.csv",
            (self.cells == OCCUPIED).astype(np.uint8),
            delimiter=",",
            fmt="%d",
        )

        state = {
            "pose": asdict(pose),
            "goal": goal,
            "route": [asdict(item) for item in route],
            "resolution_cm": self.resolution_cm,
            "origin_cm": self.origin_cm,
            "valid_readings": sum(item.distance_cm is not None for item in readings),
            "readings": [asdict(item) for item in readings],
        }
        (output_dir / "scan.json").write_text(json.dumps(state, indent=2) + "\n")
        self._render(output_dir / "map.png", pose, route=route, goal=goal)

    def _render(
        self,
        path: Path,
        pose: Pose,
        *,
        route: tuple[Pose, ...] | list[Pose] = (),
        goal: tuple[float, float] | None = None,
    ) -> None:
        colors = {
            UNKNOWN: (61, 69, 81),
            FREE: (208, 235, 219),
            OCCUPIED: (232, 74, 69),
        }
        rgb = np.empty((self.width, self.width, 3), dtype=np.uint8)
        for state, color in colors.items():
            rgb[self.cells == state] = color

        # Image coordinates grow down while world +y grows up.
        image = Image.fromarray(rgb[::-1]).resize((800, 800), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        scale = 800 / self.width

        def pixel(x_cm: float, y_cm: float) -> tuple[float, float]:
            cell_x, cell_y = self.world_to_cell(x_cm, y_cm)
            return ((cell_x + 0.5) * scale, (self.width - cell_y - 0.5) * scale)

        car_x, car_y = pixel(pose.x_cm, pose.y_cm)
        route_pixels = [pixel(item.x_cm, item.y_cm) for item in route]
        if len(route_pixels) > 1:
            draw.line(route_pixels, fill=(255, 196, 61), width=4)
        if goal is not None:
            goal_x, goal_y = pixel(*goal)
            draw.ellipse(
                (goal_x - 10, goal_y - 10, goal_x + 10, goal_y + 10),
                outline=(255, 255, 255),
                width=4,
            )
        draw.ellipse((car_x - 8, car_y - 8, car_x + 8, car_y + 8), fill=(28, 196, 232))
        heading_x = car_x + 25 * math.cos(pose.yaw_rad)
        heading_y = car_y - 25 * math.sin(pose.yaw_rad)
        draw.line((car_x, car_y, heading_x, heading_y), fill=(28, 196, 232), width=4)
        image.save(path)
