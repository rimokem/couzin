from __future__ import annotations

import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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


class PredatorStrategy(Enum):
    CLOSEST = 1
    CONFUSION = 2


class AgentConfig:
    zor: float  # Zone of Repulsion (斥力)
    zoo: float  # Zone of Orientation (整列)
    zoa: float  # Zone of Attraction (誘引)

    fov: float  # 視野角 (度)
    perception_radius: float  # 異種個体・障害物の検知半径

    speed: float  # 移動速度 (Units/s)
    turn_rate: float  # 最大回転速度 (deg/s)
    noise_sd: float  # ノイズの標準偏差 (度)

    strategy: PredatorStrategy

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
        strategy: PredatorStrategy = PredatorStrategy.CLOSEST,
    ):
        self.zor = zor
        self.zoo = zoo
        self.zoa = zoa
        self.fov = fov
        self.perception_radius = perception_radius
        self.speed = speed
        self.turn_rate = turn_rate
        self.noise_sd = noise_sd
        self.strategy = strategy


class Agent:
    id: int
    type: AgentType
    pos: NDArray[np.float64]
    v: NDArray[np.float64]
    config: AgentConfig
    target_id: int | None
    previous_visible_ids: set[int]

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
        self.previous_visible_ids = set()

    def update(self, agents: list[Agent], dt: float, boundary: float) -> None:
        self._move(agents, dt, boundary)

    def _calc_direction(
        self, agents: list[Agent], boundary: float
    ) -> NDArray[np.float64]:
        zoa_peers: list[Agent] = []
        zoo_peers: list[Agent] = []
        zor_peers: list[Agent] = []
        others: list[Agent] = []

        for agent in agents:
            if agent.id == self.id:
                continue
            distance = self._calc_distance(agent, boundary)
            if self.type != agent.type:
                if distance <= self.config.perception_radius:
                    if self._in_fov(agent, boundary):
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
            others.sort(key=lambda a: self._calc_distance(a, boundary))
            if self.type is AgentType.PREDATOR:
                if self.config.strategy == PredatorStrategy.CLOSEST:
                    target_agent = others[0]
                elif self.config.strategy == PredatorStrategy.CONFUSION:
                    current_visible_ids = {a.id for a in others}
                    new_entered_ids = current_visible_ids - self.previous_visible_ids
                    self.previous_visible_ids = current_visible_ids
                    if new_entered_ids:
                        target_id = min(new_entered_ids)
                        target_agent = next(a for a in others if a.id == target_id)
                    else:
                        if self.target_id in current_visible_ids:
                            target_agent = next(
                                a for a in others if a.id == self.target_id
                            )
                        else:
                            target_agent = others[0]

                self.target_id = target_agent.id
                diff = self._get_wrapped_diff(target_agent, boundary)
                return normalize(diff)
            else:
                diff = np.zeros((2, 1), dtype=np.float64)
                for other in others:
                    diff += self._get_wrapped_diff(other, boundary)
                    return normalize(-diff)

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
        # 数値誤差対策でclipする
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * (180.0 / np.pi)

        return angle <= (self.config.fov / 2)

    def _get_wrapped_diff(self, other: Agent, boundary: float) -> NDArray[np.float64]:
        """
        周期的境界条件を考慮して、自分から相手への最短ベクトルを計算する。
        """
        diff = other.pos - self.pos
        half_boundary = boundary / 2.0

        # X軸の補正
        if diff[0, 0] > half_boundary:
            diff[0, 0] -= boundary
        elif diff[0, 0] < -half_boundary:
            diff[0, 0] += boundary

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

    def _remove_eaten_prey(self) -> None:
        predators = [a for a in self.agents if a.type == AgentType.PREDATOR]
        surviving_prey_map = {a.id: a for a in self.agents if a.type == AgentType.PREY}
        eaten_prey_ids = set()

        for predator in predators:
            if predator.target_id is None:
                continue

            target_prey = surviving_prey_map.get(predator.target_id)
            if target_prey is None or target_prey.id in eaten_prey_ids:
                predator.target_id = None
                continue

            dist = np.linalg.norm(target_prey.pos - predator.pos)
            if dist <= self.config.catch_radius:
                eaten_prey_ids.add(target_prey.id)
                predator.target_id = None

        self.agents = [
            a
            for a in self.agents
            if (a.type == AgentType.PREDATOR) or (a.id not in eaten_prey_ids)
        ]


# --- 新しく追加した計測用関数 ---


def run_headless_simulation(config: SimulationConfig) -> float | None:
    """
    動画生成を行わずにシミュレーションを実行し、
    「最初に捕食者がターゲットをロックしてから、獲物が半減するまでの時間」
    を返します。条件を満たさない場合は None を返します。
    """
    sim = Simulation(config)

    first_lock_time: float | None = None
    initial_prey_count = config.prey_count

    max_steps = int(config.total_time / config.dt)

    for step in range(max_steps):
        current_time = step * config.dt

        sim.step()

        # 捕食者を取得（シミュレーションでは複数捕食者も想定されていますが、ここでは最初の1体または全体をチェック）
        predators = [a for a in sim.agents if a.type == AgentType.PREDATOR]

        # 1. 最初のロックオン時間を記録
        if first_lock_time is None:
            # いずれかの捕食者がターゲットを持っていればロックオンとみなす
            for p in predators:
                if p.target_id is not None:
                    first_lock_time = current_time
                    break

        # 2. 獲物の数をチェック
        current_prey_count = len([a for a in sim.agents if a.type == AgentType.PREY])

        if first_lock_time is not None:
            if current_prey_count <= initial_prey_count * 0.8:
                # 完了：経過時間を返す
                return current_time - first_lock_time

    # 時間切れで達成できなかった場合
    return None


def benchmark(config: SimulationConfig, trials: int = 10):
    """
    ProcessPoolExecutorを使ってシミュレーションを並列実行します。
    """
    results = []
    max_workers = 4

    print(f"--- Benchmark Start: {trials} trials (Parallel: {max_workers} cores) ---")
    start_cpu_time = time.time()

    # ProcessPoolExecutorで並列処理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # すべてのタスクを登録
        futures = [
            executor.submit(run_headless_simulation, config) for _ in range(trials)
        ]

        # 完了したものから順次処理（プログレス表示用）
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            duration = future.result()

            print(f"Trial {completed_count}/{trials} finished.", end=" ")
            if duration is not None:
                results.append(duration)
                print(f"Result: {duration:.2f}s")
            else:
                print("Result: Failed (Timeout)")

    elapsed_cpu_time = time.time() - start_cpu_time
    print(f"\n--- Benchmark Finished in {elapsed_cpu_time:.2f}s ---")

    if results:
        avg_time = statistics.mean(results)
        if len(results) > 1:
            stdev_time = statistics.stdev(results)
        else:
            stdev_time = 0.0

        print(f"Success Rate: {len(results)}/{trials}")
        print(f"Average Time to 80% Population (after 1st lock): {avg_time:.2f}s")
        print(f"Standard Deviation: {stdev_time:.2f}s")
        print(f"Min: {min(results):.2f}s, Max: {max(results):.2f}s")
    else:
        print(
            "No trials successfully reduced prey population to half within total_time."
        )


def run_visual_simulation(config: SimulationConfig):
    """
    既存の動画表示用ロジック
    """
    sim = Simulation(config=config)
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        sim.step()

        ax.cla()
        ax.set_xlim(0, config.boundary_size)
        ax.set_ylim(0, config.boundary_size)
        ax.set_aspect("equal")
        ax.set_title(f"Time: {frame * config.dt:.1f}s")

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

        if prey_pos:
            prey_pos = np.array(prey_pos)
            prey_v = np.array(prey_v)
            ax.scatter(prey_pos[:, 0], prey_pos[:, 1], c="blue", s=10, label="Prey")
            ax.quiver(
                prey_pos[:, 0],
                prey_pos[:, 1],
                prey_v[:, 0],
                prey_v[:, 1],
                color="blue",
                alpha=0.3,
                width=0.003,
            )

        if pred_pos:
            pred_pos = np.array(pred_pos)
            pred_v = np.array(pred_v)
            ax.scatter(
                pred_pos[:, 0],
                pred_pos[:, 1],
                c="red",
                s=30,
                label="Predator",
            )
            catch_circle = plt.Circle(
                (pred_pos[0, 0], pred_pos[0, 1]),
                config.catch_radius,
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

    _ = FuncAnimation(
        fig,
        update,
        frames=int(config.total_time / config.dt),
        interval=50,
        repeat=False,
    )
    plt.show()


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
    )
    predator_config = AgentConfig(
        zor=2.0,
        zoo=10.0,
        zoa=20.0,
        fov=180.0,
        perception_radius=20.0,
        speed=7.0,
        turn_rate=60.0,
        noise_sd=3.0,
        strategy=PredatorStrategy.CLOSEST,
    )

    # シミュレーション設定
    # ※ ベンチマーク時は total_time 内に終わらないと None になります
    sim_config = SimulationConfig(
        dt=0.1,
        total_time=1000.0,  # 時間切れを防ぐため少し長めに設定
        boundary_size=100.0,
        prey_config=prey_config,
        predator_config=predator_config,
        prey_count=50,
        predator_count=1,
        cathch_radius=1.0,
    )

    # モード選択
    print("Select Mode:")
    print("1: Run Visualization (Original)")
    print("2: Run Benchmark (Measure time to reduce population by half)")

    # ユーザー入力を待つか、ここで直接指定するか選択してください。
    # ここでは例として入力を求めます。
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        run_visual_simulation(sim_config)
    elif mode == "2":
        try:
            trials = int(input("Enter number of trials (e.g., 10): ").strip())
        except ValueError:
            trials = 10
        benchmark(sim_config, trials=trials)
    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
