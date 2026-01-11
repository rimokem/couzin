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
    wall_margin: float  # 壁からの距離の閾値

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
        wall_margin: float,
        speed: float,
        turn_rate: float,
        noise_sd: float,
    ):
        self.zor = zor
        self.zoo = zoo
        self.zoa = zoa
        self.fov = fov
        self.perception_radius = perception_radius
        self.wall_margin = wall_margin
        self.speed = speed
        self.turn_rate = turn_rate
        self.noise_sd = noise_sd


class Agent:
    id: int
    type: AgentType
    pos: NDArray[np.float64]
    v: NDArray[np.float64]
    config: AgentConfig
    target_id: int | None

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
        self.target_id = None

    def update(self, agents: list[Agent], dt: float, boundary: float) -> None:
        self._move(agents, dt, boundary)

    def _calc_direction(
        self, agents: list[Agent], boundary: float
    ) -> NDArray[np.float64]:
        zoa_peers: list[Agent] = []
        zoo_peers: list[Agent] = []
        zor_peers: list[Agent] = []
        others: list[Agent] = []

        self.target_id = None
        for agent in agents:
            if agent.id == self.id:
                continue
            distance = self._calc_distance(agent, boundary)
            if self.type != agent.type:
                if distance <= self.config.perception_radius:
                    if self._in_fov(agent, boundary):
                        # others.append(agent)
                        if others:
                            if distance < self._calc_distance(others[0], boundary):
                                others = [agent]
                        else:
                            others.append(agent)
            else:
                if distance <= self.config.zor:
                    zor_peers.append(agent)
                elif distance <= self.config.zoo:
                    if self._in_fov(agent, boundary):
                        zoo_peers.append(agent)
                elif distance <= self.config.zoa:
                    if self._in_fov(agent, boundary):
                        zoa_peers.append(agent)

        if others:
            target_agent = others[0]
            if self.type is AgentType.PREDATOR:
                self.target_id = target_agent.id

            diff = np.zeros((2, 1), dtype=np.float64)
            for other in others:
                diff += self._get_wrapped_diff(other, boundary)
            if self.type is AgentType.PREY:
                return normalize(-diff)
            elif self.type is AgentType.PREDATOR:
                return normalize(diff)

        if zor_peers:
            diff = np.zeros((2, 1), dtype=np.float64)
            for peer in zor_peers:
                diff += self._get_wrapped_diff(peer, boundary)
            return normalize(-diff)

        if zoo_peers or zoa_peers:
            d_o = self.v
            for peer in zoo_peers:
                d_o += peer.v
            d_o = normalize(d_o)

            d_a = np.zeros((2, 1), dtype=np.float64)
            for peer in zoa_peers:
                d_a += self._get_wrapped_diff(peer, boundary)
            d_a = normalize(d_a)

            return normalize(d_o + d_a)

        return normalize(self.v)

    def _calc_distance(self, other: Agent, boundary: float) -> float:
        diff = self._get_wrapped_diff(other, boundary)
        return float(np.linalg.norm(diff))

    def _in_fov(self, other: Agent, boundary: float) -> bool:
        direction_to_other = normalize(self._get_wrapped_diff(other, boundary))
        current_direction = normalize(self.v)

        dot_product = np.dot(current_direction.T, direction_to_other).item()
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180.0 / np.pi)

        return angle <= (self.config.fov / 2)

    def _get_wrapped_diff(self, other: Agent, boundary: float) -> NDArray[np.float64]:
        """
        周期的境界条件を考慮して、自分から相手への最短ベクトルを計算する。
        """
        diff = other.pos - self.pos
        half_boundary = boundary / 2.0

        # X軸の補正
        # 差が半分より大きい = 反対側から回ったほうが近い
        if diff[0, 0] > half_boundary:
            diff[0, 0] -= boundary  # 右に行き過ぎなので、左側(マイナス)補正
        elif diff[0, 0] < -half_boundary:
            diff[0, 0] += boundary  # 左に行き過ぎなので、右側(プラス)補正

        # Y軸の補正
        if diff[1, 0] > half_boundary:
            diff[1, 0] -= boundary
        elif diff[1, 0] < -half_boundary:
            diff[1, 0] += boundary

        return diff

    def _add_noise(self, direction: NDArray[np.float64]) -> NDArray[np.float64]:
        noise = np.random.normal(0, 1, 2).reshape(2, 1)
        noise = noise - np.dot(noise.T, direction)
        noise = normalize(noise)

        angle_error = np.random.normal(0, np.radians(self.config.noise_sd))

        new_direction = direction * np.cos(angle_error) + noise * np.sin(angle_error)
        return normalize(new_direction)

    def _move(self, agents: list[Agent], dt: float, boundary: float) -> None:
        direction = self._calc_direction(agents, boundary)
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

        self.pos[0, 0] = self.pos[0, 0] % boundary
        self.pos[1, 0] = self.pos[1, 0] % boundary


class SimulationConfig:
    dt: float  # タイムステップ (s)
    total_time: float  # 総シミュレーション時間 (s)
    boundary_size: float
    prey_config: AgentConfig
    predator_config: AgentConfig
    prey_count: int
    predator_count: int
    catch_radius: float

    def __init__(
        self,
        dt: float,
        total_time: float,
        boundary_size: float,
        prey_config: AgentConfig,
        predator_config: AgentConfig,
        prey_count: int,
        predator_count: int,
        cathch_radius: float,
    ):
        self.dt = dt
        self.total_time = total_time
        self.boundary_size = boundary_size
        self.prey_config = prey_config
        self.predator_config = predator_config
        self.prey_count = prey_count
        self.predator_count = predator_count
        self.catch_radius = cathch_radius


class Simulation:
    agents: list[Agent]
    config: SimulationConfig

    def __init__(self, config: SimulationConfig):
        self.agents = []
        self.config = config
        self._add_agents()

    def _add_agents(self):
        for _ in range(self.config.prey_count):
            pos = (np.random.rand(2, 1) * 0.2 + 0.5) * self.config.boundary_size
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
            pos = np.random.rand(2, 1) * self.config.boundary_size * 0.25
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

        self._remove_eaten_prey()

    def _remove_eaten_prey_all(self) -> None:
        predators = [agent for agent in self.agents if agent.type == AgentType.PREDATOR]
        prey = [agent for agent in self.agents if agent.type == AgentType.PREY]
        remaining_prey = []
        for p in prey:
            eaten = False
            for predator in predators:
                distance = np.linalg.norm(predator.pos - p.pos)
                if distance <= self.config.catch_radius:
                    eaten = True
                    break
            if not eaten:
                remaining_prey.append(p)
        self.agents = predators + remaining_prey

    def _remove_eaten_prey(self) -> None:
        """
        捕食者が「現在ロックオンしている(target_id)」個体が
        射程圏内にいる場合のみ削除する。
        """
        predators = [a for a in self.agents if a.type == AgentType.PREDATOR]

        # 現在生存しているPreyの辞書を作成 (ID検索を高速化するため)
        surviving_prey_map = {a.id: a for a in self.agents if a.type == AgentType.PREY}

        # 削除対象となったPreyのIDを記録するセット
        eaten_prey_ids = set()

        for predator in predators:
            # 誰も狙っていない場合はスキップ
            if predator.target_id is None:
                continue

            # 狙っている獲物がまだ生きているか確認
            target_prey = surviving_prey_map.get(predator.target_id)

            # すでに他の捕食者に食べられている、または視界から消えている場合はスキップ
            if target_prey is None or target_prey.id in eaten_prey_ids:
                predator.target_id = None  # ターゲットロスト
                continue

            # 狙っている特定の個体との距離のみを判定
            dist = np.linalg.norm(target_prey.pos - predator.pos)

            if dist <= self.config.catch_radius:
                # 捕食成功
                eaten_prey_ids.add(target_prey.id)
                predator.target_id = None  # 食べたのでターゲット解除

        # 最終的に生き残ったPreyだけを残す
        self.agents = [
            a
            for a in self.agents
            if (a.type == AgentType.PREDATOR) or (a.id not in eaten_prey_ids)
        ]


def main():
    prey_config = AgentConfig(
        zor=1.0,
        zoo=10.0,
        zoa=20.0,
        fov=270.0,
        perception_radius=20.0,
        speed=4.0,
        turn_rate=30.0,
        noise_sd=5.0,
        wall_margin=5.0,
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
        wall_margin=5.0,
    )
    sim_config = SimulationConfig(
        dt=0.1,
        total_time=500.0,
        boundary_size=100.0,
        prey_config=prey_config,
        predator_config=predator_config,
        prey_count=50,
        predator_count=1,
        cathch_radius=1.0,
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
                s=30,
                # marker="D",
                label="Predator",
            )
            catch_circle = plt.Circle(
                (pred_pos[0, 0], pred_pos[0, 1]),
                sim_config.catch_radius,
                color="red",
                fill=False,
                linestyle="--",
                alpha=0.5,
            )
            ax.add_artist(catch_circle)
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
