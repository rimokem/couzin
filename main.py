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


# --- 計測用関数 ---


def measure_time_to_80_percent_reduction(config: SimulationConfig) -> float | None:
    """
    最初にロックオンしてから獲物が80%に減るまでの時間を計測
    """
    sim = Simulation(config)

    first_lock_time: float | None = None
    initial_prey_count = config.prey_count

    max_steps = int(config.total_time / config.dt)

    for step in range(max_steps):
        current_time = step * config.dt

        sim.step()

        predators = [a for a in sim.agents if a.type == AgentType.PREDATOR]

        if first_lock_time is None:
            for p in predators:
                if p.target_id is not None:
                    first_lock_time = current_time
                    break

        current_prey_count = len([a for a in sim.agents if a.type == AgentType.PREY])

        if first_lock_time is not None:
            if current_prey_count <= initial_prey_count * 0.8:
                return current_time - first_lock_time

    return None


def measure_time_to_first_catch(config: SimulationConfig) -> float | None:
    """
    シミュレーション開始から、最初の獲物が捕食されるまでの時間を計測します。
    """
    sim = Simulation(config)
    initial_prey_count = config.prey_count
    max_steps = int(config.total_time / config.dt)

    for step in range(max_steps):
        current_time = step * config.dt
        sim.step()

        current_prey_count = len([a for a in sim.agents if a.type == AgentType.PREY])

        # 獲物の数が初期値より減っていれば、捕食が発生したとみなす
        if current_prey_count < initial_prey_count:
            return current_time

    return None


def benchmark_population_reduction(config: SimulationConfig, trials: int = 10):
    """
    個体数が80%に減少するまでの時間をベンチマーク
    """
    results = []
    max_workers = 4

    print(
        f"--- Benchmark (Reduction to 80%): {trials} trials (Parallel: {max_workers} cores) ---"
    )
    start_cpu_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(measure_time_to_80_percent_reduction, config)
            for _ in range(trials)
        ]

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
        stdev_time = statistics.stdev(results) if len(results) > 1 else 0.0

        print(f"Success Rate: {len(results)}/{trials}")
        print(f"Average Time: {avg_time:.2f}s")
        print(f"Standard Deviation: {stdev_time:.2f}s")
        print(f"Min: {min(results):.2f}s, Max: {max(results):.2f}s")
    else:
        print("No trials successful.")


def benchmark_first_catch(config: SimulationConfig, trials: int = 10):
    """
    初回捕食までの時間をベンチマーク
    """
    results = []
    max_workers = 4

    print(f"--- Benchmark (Time to First Catch): {trials} trials ---")
    start_cpu_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(measure_time_to_first_catch, config) for _ in range(trials)
        ]

        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            duration = future.result()

            print(f"Trial {completed_count}/{trials} finished.", end=" ")
            if duration is not None:
                results.append(duration)
                print(f"Result: {duration:.2f}s")
            else:
                print("Result: Failed (Timeout or No Catch)")

    elapsed_cpu_time = time.time() - start_cpu_time
    print(f"\n--- Benchmark Finished in {elapsed_cpu_time:.2f}s ---")

    if results:
        avg_time = statistics.mean(results)
        stdev_time = statistics.stdev(results) if len(results) > 1 else 0.0

        print(f"Success Rate: {len(results)}/{trials}")
        print(f"Average Time to First Catch: {avg_time:.2f}s")
        print(f"Standard Deviation: {stdev_time:.2f}s")
        print(f"Min: {min(results):.2f}s, Max: {max(results):.2f}s")
    else:
        print("No prey was caught in any trial.")


def plot_first_catch_vs_prey_count(
    base_config: SimulationConfig,
    min_prey: int = 1,
    max_prey: int = 50,
    trials_per_count: int = 10,
):
    """
    preyの数を変化させたときの初回捕食時間の平均をプロットする
    """
    prey_counts = list(range(min_prey, max_prey + 1))
    avg_times = []
    std_times = []

    print(f"--- Plotting First Catch Time vs Prey Count ({min_prey} to {max_prey}) ---")
    print(f"Trials per prey count: {trials_per_count}\n")

    for prey_count in prey_counts:
        print(f"Testing with {prey_count} prey...", end=" ")

        # 設定を変更
        config = SimulationConfig(
            dt=base_config.dt,
            total_time=base_config.total_time,
            boundary_size=base_config.boundary_size,
            prey_config=base_config.prey_config,
            predator_config=base_config.predator_config,
            prey_count=prey_count,
            predator_count=base_config.predator_count,
            cathch_radius=base_config.catch_radius,
        )

        # 並列実行
        results = []
        max_workers = 4
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(measure_time_to_first_catch, config)
                for _ in range(trials_per_count)
            ]
            for future in as_completed(futures):
                duration = future.result()
                if duration is not None:
                    results.append(duration)

        if results:
            avg_time = statistics.mean(results)
            std_time = statistics.stdev(results) if len(results) > 1 else 0.0
            avg_times.append(avg_time)
            std_times.append(std_time)
            print(f"Avg: {avg_time:.2f}s (±{std_time:.2f}s)")
        else:
            avg_times.append(np.nan)
            std_times.append(np.nan)
            print("No successful trials")

    # グラフ描画
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        prey_counts,
        avg_times,
        yerr=std_times,
        marker="o",
        linestyle="-",
        capsize=5,
        label="Average Time to First Catch",
    )
    ax.set_xlabel("Number of Prey", fontsize=12)
    ax.set_ylabel("Time to First Catch (s)", fontsize=12)
    ax.set_title("Time to First Catch vs Prey Count", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Plotting Complete ---")


def plot_population_reduction_vs_prey_count(
    base_config: SimulationConfig,
    min_prey: int = 1,
    max_prey: int = 50,
    trials_per_count: int = 10,
):
    """
    preyの数を変化させたときの80%減少までの時間の平均をプロットする
    """
    prey_counts = list(range(min_prey, max_prey + 1))
    avg_times = []
    std_times = []

    print(
        f"--- Plotting Time to 80% Reduction vs Prey Count ({min_prey} to {max_prey}) ---"
    )
    print(f"Trials per prey count: {trials_per_count}\n")

    for prey_count in prey_counts:
        print(f"Testing with {prey_count} prey...", end=" ")

        # 設定を変更
        config = SimulationConfig(
            dt=base_config.dt,
            total_time=base_config.total_time,
            boundary_size=base_config.boundary_size,
            prey_config=base_config.prey_config,
            predator_config=base_config.predator_config,
            prey_count=prey_count,
            predator_count=base_config.predator_count,
            cathch_radius=base_config.catch_radius,
        )

        # 並列実行
        results = []
        max_workers = 4
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(measure_time_to_80_percent_reduction, config)
                for _ in range(trials_per_count)
            ]
            for future in as_completed(futures):
                duration = future.result()
                if duration is not None:
                    results.append(duration)

        if results:
            avg_time = statistics.mean(results)
            std_time = statistics.stdev(results) if len(results) > 1 else 0.0
            avg_times.append(avg_time)
            std_times.append(std_time)
            print(f"Avg: {avg_time:.2f}s (±{std_time:.2f}s)")
        else:
            avg_times.append(np.nan)
            std_times.append(np.nan)
            print("No successful trials")

    # グラフ描画
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        prey_counts,
        avg_times,
        yerr=std_times,
        marker="o",
        linestyle="-",
        capsize=5,
        label="Average Time to 80% Reduction",
    )
    ax.set_xlabel("Number of Prey", fontsize=12)
    ax.set_ylabel("Time to 80% Reduction (s)", fontsize=12)
    ax.set_title("Time to 80% Population Reduction vs Prey Count", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("\n--- Plotting Complete ---")


def run_visualization(config: SimulationConfig):
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
        strategy=PredatorStrategy.CONFUSION,
    )

    sim_config = SimulationConfig(
        dt=0.1,
        total_time=1000.0,
        boundary_size=100.0,
        prey_config=prey_config,
        predator_config=predator_config,
        prey_count=50,
        predator_count=1,
        cathch_radius=1.0,
    )

    print("Select Mode:")
    print("1: Run Visualization")
    print("2: Run Benchmark (Population Reduction to 80%)")
    print("3: Run Benchmark (Time to First Catch)")
    print("4: Plot First Catch Time vs Prey Count")
    print("5: Plot Time to 80% Reduction vs Prey Count")

    mode = input("Enter 1, 2, 3, 4, or 5: ").strip()

    if mode == "1":
        run_visualization(sim_config)
    elif mode == "2":
        try:
            trials = int(input("Enter number of trials (e.g., 10): ").strip())
        except ValueError:
            trials = 10
        benchmark_population_reduction(sim_config, trials=trials)
    elif mode == "3":
        try:
            trials = int(input("Enter number of trials (e.g., 10): ").strip())
        except ValueError:
            trials = 10
        benchmark_first_catch(sim_config, trials=trials)
    elif mode == "4":
        try:
            min_prey = int(input("Enter minimum prey count (e.g., 1): ").strip())
            max_prey = int(input("Enter maximum prey count (e.g., 50): ").strip())
            trials = int(
                input("Enter number of trials per prey count (e.g., 10): ").strip()
            )
        except ValueError:
            min_prey = 1
            max_prey = 50
            trials = 12
        plot_first_catch_vs_prey_count(sim_config, min_prey, max_prey, trials)
    elif mode == "5":
        try:
            min_prey = int(input("Enter minimum prey count (e.g., 1): ").strip())
            max_prey = int(input("Enter maximum prey count (e.g., 50): ").strip())
            trials = int(
                input("Enter number of trials per prey count (e.g., 10): ").strip()
            )
        except ValueError:
            min_prey = 1
            max_prey = 50
            trials = 12
        plot_population_reduction_vs_prey_count(sim_config, min_prey, max_prey, trials)
    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
