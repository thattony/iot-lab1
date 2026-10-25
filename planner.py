"""Hybrid A* route planner for the Ackermann-steered PiCar-X."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

from mapping import UNKNOWN, OccupancyGrid, Pose


@dataclass(frozen=True)
class Motion:
    pose: Pose
    steering_deg: float
    distance_cm: float


@dataclass
class _Node:
    pose: Pose
    parent: "_Node | None"
    steering_deg: float
    distance_cm: float
    cost: float


class HybridAStar:
    """A* over position, heading, and feasible PiCar-X steering arcs."""

    def __init__(
        self,
        grid: OccupancyGrid,
        *,
        clearance_cm: float = 18.0,
        wheelbase_cm: float = 12.0,
        step_cm: float = 8.0,
        heading_bins: int = 72,
        unknown_penalty: float = 0.35,
    ):
        self.grid = grid
        self.blocked = grid.inflated_obstacles(clearance_cm)
        self.wheelbase_cm = wheelbase_cm
        self.step_cm = step_cm
        self.heading_bins = heading_bins
        self.unknown_penalty = unknown_penalty
        self.steering_options = (-28.0, -14.0, 0.0, 14.0, 28.0)

    def _key(self, pose: Pose) -> tuple[int, int, int]:
        cell_x, cell_y = self.grid.world_to_cell(pose.x_cm, pose.y_cm)
        heading = round(
            (pose.yaw_rad % (2.0 * math.pi)) / (2.0 * math.pi / self.heading_bins)
        ) % self.heading_bins
        return cell_x, cell_y, heading

    def _valid(self, pose: Pose) -> bool:
        cell = self.grid.world_to_cell(pose.x_cm, pose.y_cm)
        return self.grid.inside(cell) and not self.blocked[cell[1], cell[0]]

    def _motion_is_valid(self, start: Pose, steering_deg: float) -> tuple[bool, Pose]:
        steering_rad = math.radians(steering_deg)
        end = start
        for index in range(1, 6):
            end = start.advance_bicycle(
                self.step_cm * index / 5.0, steering_rad, self.wheelbase_cm
            )
            if not self._valid(end):
                return False, end
        return True, end

    @staticmethod
    def _distance(pose: Pose, goal: tuple[float, float]) -> float:
        return math.hypot(pose.x_cm - goal[0], pose.y_cm - goal[1])

    def plan(
        self,
        start: Pose,
        goal: tuple[float, float],
        *,
        goal_tolerance_cm: float = 10.0,
        max_expansions: int = 100_000,
    ) -> list[Motion] | None:
        goal_pose = Pose(goal[0], goal[1])
        if not self._valid(start) or not self._valid(goal_pose):
            return None

        root = _Node(start, None, 0.0, 0.0, 0.0)
        counter = itertools.count()
        frontier: list[tuple[float, int, _Node]] = [
            (self._distance(start, goal), next(counter), root)
        ]
        best_cost = {self._key(start): 0.0}

        for _ in range(max_expansions):
            if not frontier:
                return None
            _, _, node = heapq.heappop(frontier)
            if node.cost > best_cost.get(self._key(node.pose), math.inf) + 1e-9:
                continue
            if self._distance(node.pose, goal) <= goal_tolerance_cm:
                result: list[Motion] = []
                while node.parent is not None:
                    result.append(Motion(node.pose, node.steering_deg, node.distance_cm))
                    node = node.parent
                return list(reversed(result))

            for steering_deg in self.steering_options:
                valid, end = self._motion_is_valid(node.pose, steering_deg)
                if not valid:
                    continue
                cell_x, cell_y = self.grid.world_to_cell(end.x_cm, end.y_cm)
                unknown_cost = (
                    self.step_cm * self.unknown_penalty
                    if self.grid.cells[cell_y, cell_x] == UNKNOWN
                    else 0.0
                )
                steering_cost = self.step_cm * 0.12 * abs(steering_deg) / 28.0
                steering_change_cost = 0.03 * abs(steering_deg - node.steering_deg)
                cost = node.cost + self.step_cm + unknown_cost + steering_cost + steering_change_cost
                key = self._key(end)
                if cost >= best_cost.get(key, math.inf):
                    continue
                best_cost[key] = cost
                child = _Node(end, node, steering_deg, self.step_cm, cost)
                priority = cost + self._distance(end, goal)
                heapq.heappush(frontier, (priority, next(counter), child))
        return None

