from __future__ import annotations

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from numpy.typing import NDArray


def normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class AgentType(Enum):
    PREY = 1
    PREDATOR = 2


class AgentConfig:
    zor: float  # Zone of Repulsion (斥力)
    zoo: float  # Zone of Orientation (整列)
    zoa: float  # Zone of Attraction (誘引)

    fov: float  # 視野角 (度)
    perception_radius: float  # 異種個体・障害物の検知半径

    speed: float  # 移動速度 (Units/s)
    turn_rate: float  # 最大回転速度 (deg/s)
    noise_sd: float  # ノイズの標準偏差 (度)

    def __init__(
        self,
        zor: float,
        zoo: float,
        zoa: float,
        fov: float,
        perception_radius: float,
        speed: float,
        turn_rate: float,
        noise_sd: float,
    ):
        self.zor = zor
        self.zoo = zoo
        self.zoa = zoa
        self.fov = fov
        self.perception_radius = perception_radius
        self.speed = speed
        self.turn_rate = turn_rate
        self.noise_sd = noise_sd


class Agent:
    id: int
    type: AgentType
    pos: NDArray[np.float64]
    v: NDArray[np.float64]
    config: AgentConfig

    def __init__(
        self,
        id: int,
        type: AgentType,
        pos: NDArray[np.float64],
        v: NDArray[np.float64],
        config: AgentConfig,
    ):
        self.id = id
        self.type = type
        self.pos = pos
        if self.pos.shape != (2, 1):
            raise ValueError("`pos` must be a (2,1) ndarray")

        self.v = v
        if self.v.shape != (2, 1):
            raise ValueError("`v` must be a (2,1) ndarray")
        self.config = config

    def update(self, agents: list[Agent], dt: float, boundary: float) -> None:
        self._move(agents, dt, boundary)

    def _calc_direction(self, agents: list[Agent]) -> NDArray[np.float64]:
        zoa_peers: list[Agent] = []
        zoo_peers: list[Agent] = []
        zor_peers: list[Agent] = []
        others: list[Agent] = []
        for agent in agents:
            if agent.id == self.id:
                continue
            distance = self._calc_distance(agent)
            if self.type != agent.type:
                if distance <= self.config.perception_radius:
                    if self._in_fov(agent):
                        others.append(agent)
            else:
                if distance <= self.config.zor:
                    zor_peers.append(agent)
                elif distance <= self.config.zoo:
                    if self._in_fov(agent):
                        zoo_peers.append(agent)
                elif distance <= self.config.zoa:
                    if self._in_fov(agent):
                        zoa_peers.append(agent)

        if others:
            diff = np.zeros((2, 1), dtype=np.float64)
            for other in others:
                diff += other.pos - self.pos
            if self.type is AgentType.PREY:
                return normalize(-diff)
            elif self.type is AgentType.PREDATOR:
                return normalize(diff)

        if zor_peers:
            diff = np.zeros((2, 1), dtype=np.float64)
            for peer in zor_peers:
                diff += peer.pos - self.pos
            return normalize(-diff)

        if zoo_peers or zoa_peers:
            d_o = self.v
            for peer in zoo_peers:
                d_o += peer.v
            d_o = normalize(d_o)

            d_a = np.zeros((2, 1), dtype=np.float64)
            for peer in zoa_peers:
                d_a += peer.pos - self.pos
            d_a = normalize(d_a)

            if zoa_peers:
                return normalize(d_o + d_a)
            else:
                return d_o

        return normalize(self.v)

    def _calc_distance(self, other: Agent) -> float:
        diff = other.pos - self.pos
        return float(np.linalg.norm(diff))

    def _in_fov(self, other: Agent) -> bool:
        direction_to_other = normalize(other.pos - self.pos)
        current_direction = normalize(self.v)

        dot_product = np.dot(current_direction.T, direction_to_other).item()
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180.0 / np.pi)

        return angle <= (self.config.fov / 2)

    def _add_noise(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        noise = np.random.normal(0, 1, 2).reshape(2, 1)
        noise = noise - np.dot(noise.T, direction)
        noise = normalize(noise)

        angle_error = np.random.normal(0, np.radians(self.config.noise_sd))

        new_direction = direction * np.cos(angle_error) + noise * np.sin(angle_error)
        return normalize(new_direction)

    def _move(self, agents: list[Agent], dt: float, boundary: float) -> None:
        direction = self._calc_direction(agents)
        direction = self._add_noise(direction)
        # 方向がturn_rateを超えている場合、directionを調整
        current_direction = normalize(self.v)
        dot_product = np.dot(current_direction.T, direction).item()
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180.0 / np.pi)
        if angle > self.config.turn_rate * dt:
            direction = current_direction * np.cos(
                np.deg2rad(self.config.turn_rate * dt)
            ) + normalize(direction - current_direction * dot_product) * np.sin(
                np.deg2rad(self.config.turn_rate * dt)
            )
            direction = normalize(direction)
        self.v = direction * self.config.speed
        self.pos += self.v * dt

        if self.pos[0, 0] < 0:
            self.pos[0, 0] = 0
            self.v[0, 0] *= -1
        elif self.pos[0, 0] > boundary:
            self.pos[0, 0] = boundary
            self.v[0, 0] *= -1

        if self.pos[1, 0] < 0:
            self.pos[1, 0] = 0
            self.v[1, 0] *= -1
        elif self.pos[1, 0] > boundary:
            self.pos[1, 0] = boundary
            self.v[1, 0] *= -1


class SimulationConfig:
    dt: float  # タイムステップ (s)
    total_time: float  # 総シミュレーション時間 (s)
    boundary_size: float
    prey_config: AgentConfig
    predator_config: AgentConfig
    prey_count: int
    predator_count: int

    def __init__(
        self,
        dt: float,
        total_time: float,
        boundary_size: float,
        prey_config: AgentConfig,
        predator_config: AgentConfig,
        prey_count: int,
        predator_count: int,
    ):
        self.dt = dt
        self.total_time = total_time
        self.boundary_size = boundary_size
        self.prey_config = prey_config
        self.predator_config = predator_config
        self.prey_count = prey_count
        self.predator_count = predator_count


class Simulation:
    agents: list[Agent]
    config: SimulationConfig

    def __init__(self, config: SimulationConfig):
        self.agents = []
        self.config = config
        self._add_agents()

    def _add_agents(self):
        for _ in range(self.config.prey_count):
            pos = np.random.rand(2, 1) * self.config.boundary_size * 0.25
            angle = np.random.rand() * 2 * np.pi
            v = (
                np.array([[np.cos(angle)], [np.sin(angle)]])
                * self.config.prey_config.speed
            )
            agent = Agent(
                id=len(self.agents),
                type=AgentType.PREY,
                pos=pos,
                v=v,
                config=self.config.prey_config,
            )
            self.agents.append(agent)
        for _ in range(self.config.predator_count):
            pos = (np.random.rand(2, 1) * 0.3 + 0.7) * self.config.boundary_size
            angle = np.random.rand() * 2 * np.pi
            v = (
                np.array([[np.cos(angle)], [np.sin(angle)]])
                * self.config.predator_config.speed
            )
            agent = Agent(
                id=len(self.agents),
                type=AgentType.PREDATOR,
                pos=pos,
                v=v,
                config=self.config.predator_config,
            )
            self.agents.append(agent)

    def step(self) -> None:
        dt = self.config.dt
        boundary = self.config.boundary_size
        for i, agent in enumerate(self.agents):
            agent.update(self.agents, dt, boundary)


def main():
    prey_config = AgentConfig(
        zor=1.0,
        zoo=5.0,
        zoa=10.0,
        fov=270.0,
        perception_radius=10.0,
        speed=4.0,
        turn_rate=30.0,
        noise_sd=5.0,
    )
    predator_config = AgentConfig(
        zor=2.0,
        zoo=10.0,
        zoa=20.0,
        fov=200.0,
        perception_radius=20.0,
        speed=4.5,
        turn_rate=60.0,
        noise_sd=3.0,
    )
    sim_config = SimulationConfig(
        dt=0.1,
        total_time=1000.0,
        boundary_size=100.0,
        prey_config=prey_config,
        predator_config=predator_config,
        prey_count=30,
        predator_count=1,
    )
    sim = Simulation(config=sim_config)
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        sim.step()

        ax.cla()  # 前のフレームをクリア
        ax.set_xlim(0, sim_config.boundary_size)
        ax.set_ylim(0, sim_config.boundary_size)
        ax.set_aspect("equal")
        ax.set_title(f"Time: {frame * sim_config.dt:.1f}s")

        # データの抽出
        prey_pos = []
        prey_v = []
        pred_pos = []
        pred_v = []

        for agent in sim.agents:
            if agent.type == AgentType.PREY:
                prey_pos.append(agent.pos.flatten())
                prey_v.append(agent.v.flatten())
            else:
                pred_pos.append(agent.pos.flatten())
                pred_v.append(agent.v.flatten())

        # Preyの描画 (青い矢印)
        if prey_pos:
            prey_pos = np.array(prey_pos)
            prey_v = np.array(prey_v)
            # 位置
            ax.scatter(prey_pos[:, 0], prey_pos[:, 1], c="blue", s=10, label="Prey")
            # 向き（矢印）
            ax.quiver(
                prey_pos[:, 0],
                prey_pos[:, 1],
                prey_v[:, 0],
                prey_v[:, 1],
                color="blue",
                alpha=0.3,
                width=0.003,
            )

        # Predatorの描画 (赤い大きな点と矢印)
        if pred_pos:
            pred_pos = np.array(pred_pos)
            pred_v = np.array(pred_v)
            ax.scatter(
                pred_pos[:, 0],
                pred_pos[:, 1],
                c="red",
                s=50,
                marker="D",
                label="Predator",
            )
            ax.quiver(
                pred_pos[:, 0],
                pred_pos[:, 1],
                pred_v[:, 0],
                pred_v[:, 1],
                color="red",
                width=0.005,
            )

        ax.legend(loc="upper right")

    # 4. アニメーション実行
    _ = FuncAnimation(
        fig,
        update,
        frames=int(sim_config.total_time / sim_config.dt),
        interval=50,
        repeat=False,
    )
    plt.show()


if __name__ == "__main__":
    main()
