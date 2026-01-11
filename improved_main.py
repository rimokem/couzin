from __future__ import annotations

import statistics
import time
from enum import IntEnum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# --- 設定クラス (変更なし) ---
class AgentConfig:
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
        self.fov = np.radians(fov)  # ラジアンに変換して保持
        self.perception_radius = perception_radius
        self.wall_margin = wall_margin
        self.speed = speed
        self.turn_rate = np.radians(turn_rate)  # ラジアンに変換
        self.noise_sd = np.radians(noise_sd)  # ラジアンに変換


class SimulationConfig:
    def __init__(
        self,
        dt: float,
        total_time: float,
        boundary_size: float,
        prey_config: AgentConfig,
        predator_config: AgentConfig,
        prey_count: int,
        predator_count: int,
        catch_radius: float,
    ):
        self.dt = dt
        self.total_time = total_time
        self.boundary_size = boundary_size
        self.prey_config = prey_config
        self.predator_config = predator_config
        self.prey_count = prey_count
        self.predator_count = predator_count
        self.catch_radius = catch_radius


class AgentType(IntEnum):
    PREY = 0
    PREDATOR = 1


# --- 高速化されたシミュレーションクラス ---
class VectorizedSimulation:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.boundary = np.array([config.boundary_size, config.boundary_size])

        # エージェント総数
        self.n_prey = config.prey_count
        self.n_predator = config.predator_count
        self.n_total = self.n_prey + self.n_predator

        # 配列の初期化 (N, 2)
        # 0 ~ n_prey-1 : Prey
        # n_prey ~ end : Predator
        self.pos = np.zeros((self.n_total, 2))
        self.vel = np.zeros((self.n_total, 2))
        self.types = np.zeros(self.n_total, dtype=int)
        self.ids = np.arange(self.n_total)  # 固定ID

        # パラメータ配列の準備（ブロードキャスト用）
        self._init_arrays()

        # 捕食者のターゲット管理 (Predator数分の配列, 初期値 -1)
        self.predator_target_ids = np.full(self.n_predator, -1, dtype=int)

    def _init_arrays(self):
        # Preyの初期化
        self.pos[: self.n_prey] = (
            np.random.rand(self.n_prey, 2) * self.cfg.boundary_size
        )
        angles = np.random.rand(self.n_prey) * 2 * np.pi
        self.vel[: self.n_prey] = (
            np.column_stack((np.cos(angles), np.sin(angles)))
            * self.cfg.prey_config.speed
        )
        self.types[: self.n_prey] = AgentType.PREY

        # Predatorの初期化
        self.pos[self.n_prey :] = (
            np.random.rand(self.n_predator, 2) * self.cfg.boundary_size
        )
        angles = np.random.rand(self.n_predator) * 2 * np.pi
        self.vel[self.n_prey :] = (
            np.column_stack((np.cos(angles), np.sin(angles)))
            * self.cfg.predator_config.speed
        )
        self.types[self.n_prey :] = AgentType.PREDATOR

        # 設定値を配列化して計算時に参照しやすくする
        # shape: (N, 1)
        self.speeds = np.zeros((self.n_total, 1))
        self.speeds[: self.n_prey] = self.cfg.prey_config.speed
        self.speeds[self.n_prey :] = self.cfg.predator_config.speed

        self.turn_rates = np.zeros((self.n_total, 1))
        self.turn_rates[: self.n_prey] = self.cfg.prey_config.turn_rate
        self.turn_rates[self.n_prey :] = self.cfg.predator_config.turn_rate

        self.noises = np.zeros((self.n_total, 1))
        self.noises[: self.n_prey] = self.cfg.prey_config.noise_sd
        self.noises[self.n_prey :] = self.cfg.predator_config.noise_sd

        # 距離閾値なども配列化
        self.zors = np.zeros((self.n_total, 1))
        self.zors[: self.n_prey] = self.cfg.prey_config.zor
        self.zors[self.n_prey :] = self.cfg.predator_config.zor
        self.zoos = np.zeros((self.n_total, 1))
        self.zoos[: self.n_prey] = self.cfg.prey_config.zoo
        self.zoos[self.n_prey :] = self.cfg.predator_config.zoo
        self.zoas = np.zeros((self.n_total, 1))
        self.zoas[: self.n_prey] = self.cfg.prey_config.zoa
        self.zoas[self.n_prey :] = self.cfg.predator_config.zoa
        self.perceptions = np.zeros((self.n_total, 1))
        self.perceptions[: self.n_prey] = self.cfg.prey_config.perception_radius
        self.perceptions[self.n_prey :] = self.cfg.predator_config.perception_radius
        self.fovs = np.zeros((self.n_total, 1))
        self.fovs[: self.n_prey] = self.cfg.prey_config.fov
        self.fovs[self.n_prey :] = self.cfg.predator_config.fov

    def step(self):
        N = self.n_total
        if N == 0:
            return

        # 1. 全ペアの相対ベクトルと距離を一括計算 (Toroidal boundary)
        # diff[i, j] = pos[j] - pos[i] (iから見たjのベクトル)
        # shape: (N, N, 2)
        diff = self.pos[None, :, :] - self.pos[:, None, :]

        # 境界条件 (diff > L/2 なら -L, diff < -L/2 なら +L)
        diff = (diff + self.boundary / 2) % self.boundary - self.boundary / 2

        # 距離計算 (N, N)
        dist = np.linalg.norm(diff, axis=2)

        # 自分自身との距離は無限大にしておく（計算除外のため）
        np.fill_diagonal(dist, np.inf)

        # 2. 視野角 (FOV) チェック
        # 現在の速度ベクトルを正規化
        vel_norm = np.linalg.norm(self.vel, axis=1, keepdims=True)
        vel_dir = np.divide(
            self.vel, vel_norm, out=np.zeros_like(self.vel), where=vel_norm != 0
        )

        # 相手への方向ベクトル
        # diff_norm[i, j] = normalized(diff[i, j])
        dist_expanded = dist[:, :, None]
        dir_to_neighbor = np.divide(
            diff, dist_expanded, out=np.zeros_like(diff), where=dist_expanded != 0
        )

        # 内積 (dot product) -> cos theta
        # (N, 1, 2) * (N, N, 2) -> (N, N)
        cos_angle = np.einsum("ni,nji->nj", vel_dir, dir_to_neighbor)
        # クリップしてacos
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # マスク作成: FOV内かどうか
        in_fov = angle <= (self.fovs / 2)  # broadcasting (N, 1) vs (N, N)

        # 3. マスク作成: 各種条件
        # 同種・異種マスク
        same_type = self.types[:, None] == self.types[None, :]
        diff_type = ~same_type

        # 距離条件マスク
        in_perception = dist <= self.perceptions
        in_zor = (dist <= self.zors) & same_type
        in_zoo = (dist <= self.zoos) & same_type & in_fov
        in_zoa = (dist <= self.zoas) & same_type & in_fov

        # 異種（敵/獲物）検知
        detected_enemies = diff_type & in_perception & in_fov

        # --- 方向ベクトルの決定 ---
        # 最終的なターゲット方向 (N, 2)
        target_dirs = np.zeros_like(self.vel)
        has_target = np.zeros(N, dtype=bool)

        # A. 異種に対する反応 (最優先)
        # detected_enemiesマスクを持つ行があるか
        # 捕食者: 一番近いPreyを追う / Prey: 一番近いPredatorから逃げる

        # Preyの処理 (逃げる)
        # 敵が見えているPreyのインデックス
        prey_indices = np.where(self.types == AgentType.PREY)[0]
        # そのPreyが見ている敵のマスク
        prey_enemy_mask = detected_enemies[prey_indices]

        # 敵が見えている場合のみ計算
        # np.any(prey_enemy_mask, axis=1) -> 各Preyが敵を見ているか
        flee_active_idx = prey_indices[np.any(prey_enemy_mask, axis=1)]

        if len(flee_active_idx) > 0:
            # 見えている敵のうち、最も近いものを見つける
            # 距離行列から、敵以外を無限大にしてargminをとる
            dists_for_prey = dist[flee_active_idx].copy()
            # 敵でない、またはFOV外の場所を無限大に
            mask_invalid = ~detected_enemies[flee_active_idx]
            dists_for_prey[mask_invalid] = np.inf

            nearest_enemy_idx = np.argmin(dists_for_prey, axis=1)

            # 逃げるベクトル = -(相手へのベクトル)
            # diff[i, nearest_j]
            flee_vecs = -diff[flee_active_idx, nearest_enemy_idx]

            # 正規化
            norms = np.linalg.norm(flee_vecs, axis=1, keepdims=True)
            flee_vecs = np.divide(
                flee_vecs, norms, out=np.zeros_like(flee_vecs), where=norms != 0
            )

            target_dirs[flee_active_idx] = flee_vecs
            has_target[flee_active_idx] = True

        # Predatorの処理 (追う)
        pred_indices = np.where(self.types == AgentType.PREDATOR)[0]
        pred_prey_mask = detected_enemies[pred_indices]

        # 獲物が見えているPredator
        chase_active_idx = pred_indices[np.any(pred_prey_mask, axis=1)]

        # ロック状態のリセット確認
        # (簡単のため、今回は毎フレーム「最も近い獲物」をロック対象とする仕様にします)
        # ベンチマーク用に「ターゲットを持っているか」は target_ids で管理
        self.predator_target_ids[:] = -1

        if len(chase_active_idx) > 0:
            dists_for_pred = dist[chase_active_idx].copy()
            mask_invalid = ~detected_enemies[chase_active_idx]
            dists_for_pred[mask_invalid] = np.inf

            nearest_prey_idx = np.argmin(dists_for_pred, axis=1)
            # ローカルインデックスからグローバルIDへ変換が必要だが、
            # diff行列は(N,N)なのでそのままindexとして使える

            # ターゲットIDを記録 (グローバルID = index)
            # chase_active_idx は pred_indices のサブセット。対応付けが必要
            # pred_indices内での相対位置を計算
            rel_idx = np.searchsorted(pred_indices, chase_active_idx)
            self.predator_target_ids[rel_idx] = self.ids[nearest_prey_idx]

            chase_vecs = diff[chase_active_idx, nearest_prey_idx]

            norms = np.linalg.norm(chase_vecs, axis=1, keepdims=True)
            chase_vecs = np.divide(
                chase_vecs, norms, out=np.zeros_like(chase_vecs), where=norms != 0
            )

            target_dirs[chase_active_idx] = chase_vecs
            has_target[chase_active_idx] = True

        # B. 群れのルール (異種への反応がない個体のみ適用)
        # 計算対象: has_target == False の個体
        flock_indices = np.where(~has_target)[0]

        if len(flock_indices) > 0:
            # --- Separation (ZOR) ---
            # 斥力: 近すぎる仲間から離れる (-diff の平均)
            zor_mask = in_zor[flock_indices]
            # 各個体について、zor内に仲間がいるか
            has_zor = np.any(zor_mask, axis=1)

            # zor有効な個体のindex
            zor_active = flock_indices[has_zor]

            if len(zor_active) > 0:
                # diff[zor_active] shape: (M, N, 2)
                # mask: (M, N) -> (M, N, 1)
                mask_expanded = in_zor[zor_active][:, :, None]
                # マスクされたdiffの和を取る
                # -diff なので (pos[i] - pos[j])
                repulsion = -np.sum(diff[zor_active] * mask_expanded, axis=1)

                # 正規化
                norms = np.linalg.norm(repulsion, axis=1, keepdims=True)
                repulsion = np.divide(
                    repulsion, norms, out=np.zeros_like(repulsion), where=norms != 0
                )

                target_dirs[zor_active] = repulsion
                has_target[zor_active] = True

        # --- Alignment (ZOO) & Cohesion (ZOA) ---
        # まだターゲットが決まっていない個体 (ZORもなかった個体)
        flock_indices = np.where(~has_target)[0]
        if len(flock_indices) > 0:
            # ZOO: 仲間の速度の平均
            zoo_mask = in_zoo[flock_indices][:, :, None]  # (K, N, 1)
            zoo_sum = np.sum(self.vel[None, :, :] * zoo_mask, axis=1)
            # 自分の速度も足す（元ロジック準拠）
            zoo_sum += self.vel[flock_indices]

            zoo_vec = np.zeros_like(zoo_sum)
            norms = np.linalg.norm(zoo_sum, axis=1, keepdims=True)
            zoo_vec = np.divide(
                zoo_sum, norms, out=np.zeros_like(zoo_vec), where=norms != 0
            )

            # ZOA: 仲間へのベクトルの平均
            zoa_mask = in_zoa[flock_indices][:, :, None]
            zoa_sum = np.sum(diff[flock_indices] * zoa_mask, axis=1)

            zoa_vec = np.zeros_like(zoa_sum)
            norms = np.linalg.norm(zoa_sum, axis=1, keepdims=True)
            zoa_vec = np.divide(
                zoa_sum, norms, out=np.zeros_like(zoa_vec), where=norms != 0
            )

            # ZOOとZOAがある場合は合成
            # どちらもマスクが空(0ベクトル)なら影響なし
            combined = zoo_vec + zoa_vec
            norms = np.linalg.norm(combined, axis=1, keepdims=True)
            final_vec = np.divide(
                combined, norms, out=np.zeros_like(combined), where=norms != 0
            )

            # どちらもゼロなら現在の速度維持
            zero_mask = norms.flatten() == 0
            final_vec[zero_mask] = vel_dir[flock_indices][zero_mask]

            target_dirs[flock_indices] = final_vec
            has_target[flock_indices] = True

        # ターゲットが決まらなかった個体は現在の進行方向
        no_target = ~has_target
        target_dirs[no_target] = vel_dir[no_target]

        # 4. ノイズ付加と回転制限、位置更新
        # ノイズ
        noise_vec = np.random.normal(0, 1, (N, 2))
        # 進行方向に対して垂直成分のみにする (Gram-Schmidt)
        # noise = noise - dot(noise, target) * target
        dot_nt = np.sum(noise_vec * target_dirs, axis=1, keepdims=True)
        noise_ortho = noise_vec - dot_nt * target_dirs
        norm_noise = np.linalg.norm(noise_ortho, axis=1, keepdims=True)
        noise_ortho = np.divide(
            noise_ortho,
            norm_noise,
            out=np.zeros_like(noise_ortho),
            where=norm_noise != 0,
        )

        # 角度誤差
        angle_err = np.random.normal(0, self.noises).flatten()  # (N,)
        sin_err = np.sin(angle_err)[:, None]
        cos_err = np.cos(angle_err)[:, None]

        noisy_target = target_dirs * cos_err + noise_ortho * sin_err

        # 回転制限 (Current Vel vs Noisy Target)
        # 内積
        dot_prod = np.sum(vel_dir * noisy_target, axis=1)
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        diff_angle = np.arccos(dot_prod)

        max_turn = self.turn_rates.flatten() * self.cfg.dt

        # 制限を超える場合のみ補間 (Slerp的な線形補間で近似)
        # 回転軸などは2Dなので、単に角度制限内でターゲットに寄せればよい
        # ここでは簡易的に「今のベクトル」と「目標」を重み付けして正規化する手法をとる
        # あるいは、回転行列を使う。

        # 回転が必要な個体
        turn_mask = diff_angle > max_turn

        final_dirs = noisy_target.copy()

        if np.any(turn_mask):
            # 制限を超えた分だけ回す処理はベクトル演算だと少し複雑。
            # 簡易実装: 現在のベクトルを max_turn だけ目標方向に回す
            # 2Dの外積 (z成分) で回転方向判定: A x B = AxBy - AyBx
            cross_z = (
                vel_dir[:, 0] * noisy_target[:, 1] - vel_dir[:, 1] * noisy_target[:, 0]
            )
            sign = np.sign(cross_z)  # +1: 左回り, -1: 右回り

            # 回転行列で現在の速度を回す
            # Rot(theta) = [[cos, -sin], [sin, cos]]
            theta = max_turn * sign
            c = np.cos(theta)
            s = np.sin(theta)

            # 回転後のベクトル (x', y')
            # x' = x*c - y*s
            # y' = x*s + y*c
            # ただし mask 箇所のみ
            vx = vel_dir[turn_mask, 0]
            vy = vel_dir[turn_mask, 1]

            nx = vx * c[turn_mask] - vy * s[turn_mask]
            ny = vx * s[turn_mask] + vy * c[turn_mask]

            final_dirs[turn_mask] = np.column_stack((nx, ny))

        # 速度更新
        self.vel = final_dirs * self.speeds

        # 位置更新
        self.pos += self.vel * self.cfg.dt
        self.pos %= self.boundary  # Wrap

        # 5. 捕食判定と削除
        self._handle_eating(dist)

    def _handle_eating(self, dist_matrix):
        # Predator (rows) vs Prey (cols)
        # dist_matrix: (N, N)

        pred_indices = np.where(self.types == AgentType.PREDATOR)[0]
        prey_indices = np.where(self.types == AgentType.PREY)[0]

        if len(pred_indices) == 0 or len(prey_indices) == 0:
            return

        # ターゲットを持っているPredatorのみ判定 (ロックオン仕様に準拠する場合)
        # 今回の target_ids には index が入っている

        eaten_prey_indices = []

        # Vectorized check is tricky if based on specific target ID.
        # But we can check distance to the *locked* target.

        for i, p_idx in enumerate(pred_indices):
            target_idx = self.predator_target_ids[i]
            if target_idx != -1:
                # ターゲットとの距離
                d = dist_matrix[p_idx, target_idx]
                if d <= self.cfg.catch_radius:
                    eaten_prey_indices.append(target_idx)

        if eaten_prey_indices:
            eaten_prey_indices = list(set(eaten_prey_indices))  # 重複排除
            # 削除処理: マスクを作成して配列を再構築
            keep_mask = np.ones(self.n_total, dtype=bool)
            keep_mask[eaten_prey_indices] = False

            self.pos = self.pos[keep_mask]
            self.vel = self.vel[keep_mask]
            self.types = self.types[keep_mask]
            self.ids = self.ids[keep_mask]
            self.speeds = self.speeds[keep_mask]
            self.turn_rates = self.turn_rates[keep_mask]
            self.noises = self.noises[keep_mask]

            # パラメータ配列も縮小
            self.zors = self.zors[keep_mask]
            self.zoos = self.zoos[keep_mask]
            self.zoas = self.zoas[keep_mask]
            self.perceptions = self.perceptions[keep_mask]
            self.fovs = self.fovs[keep_mask]

            # カウント更新
            self.n_prey = np.sum(self.types == AgentType.PREY)
            self.n_predator = np.sum(self.types == AgentType.PREDATOR)
            self.n_total = len(self.types)

            # Predatorターゲット配列の整合性が崩れるのでリセット
            # (次ステップで再検索するので問題なし)
            self.predator_target_ids = np.full(self.n_predator, -1, dtype=int)


# --- 計測・実行用関数 ---


def run_benchmark_vectorized(config: SimulationConfig, trials: int = 10):
    """高速化版シミュレーションでベンチマーク"""
    results = []
    print(f"--- Vectorized Benchmark: {trials} trials ---")

    start_cpu = time.time()

    for i in range(trials):
        sim = VectorizedSimulation(config)

        initial_prey = sim.n_prey
        first_lock_time = None
        duration = None

        max_steps = int(config.total_time / config.dt)

        for step in range(max_steps):
            current_time = step * config.dt
            sim.step()

            # ロック判定: ターゲットIDを持っているPredatorがいるか
            if first_lock_time is None:
                if np.any(sim.predator_target_ids != -1):
                    first_lock_time = current_time

            # 半減判定
            if first_lock_time is not None:
                if sim.n_prey <= initial_prey / 2:
                    duration = current_time - first_lock_time
                    break

        if duration is not None:
            results.append(duration)
            print(f"Trial {i + 1}: {duration:.2f}s")
        else:
            print(f"Trial {i + 1}: Failed/Timeout")

    total_time = time.time() - start_cpu
    print(f"\n--- Benchmark Finished in {total_time:.2f}s ---")

    if results:
        print(f"Average: {statistics.mean(results):.2f}s")
        if len(results) > 1:
            print(f"Stdev: {statistics.stdev(results):.2f}s")
    else:
        print("No successful trials.")


def run_visual_vectorized(config: SimulationConfig):
    sim = VectorizedSimulation(config)
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame):
        sim.step()
        ax.cla()
        ax.set_xlim(0, config.boundary_size)
        ax.set_ylim(0, config.boundary_size)
        ax.set_aspect("equal")
        ax.set_title(
            f"Vectorized Sim | Time: {frame * config.dt:.1f}s | Prey: {sim.n_prey}"
        )

        if sim.n_total > 0:
            # 配列から直接プロット
            prey_mask = sim.types == AgentType.PREY
            pred_mask = sim.types == AgentType.PREDATOR

            if np.any(prey_mask):
                pos_prey = sim.pos[prey_mask]
                vel_prey = sim.vel[prey_mask]
                ax.scatter(pos_prey[:, 0], pos_prey[:, 1], c="blue", s=10, label="Prey")
                # 矢印は重いので間引くか、軽量化して描画
                ax.quiver(
                    pos_prey[:, 0],
                    pos_prey[:, 1],
                    vel_prey[:, 0],
                    vel_prey[:, 1],
                    color="blue",
                    alpha=0.3,
                    width=0.003,
                )

            if np.any(pred_mask):
                pos_pred = sim.pos[pred_mask]
                vel_pred = sim.vel[pred_mask]
                ax.scatter(
                    pos_pred[:, 0], pos_pred[:, 1], c="red", s=30, label="Predator"
                )

                # 捕獲範囲
                for p in pos_pred:
                    circle = plt.Circle(
                        (p[0], p[1]),
                        config.catch_radius,
                        color="red",
                        fill=False,
                        linestyle="--",
                        alpha=0.5,
                    )
                    ax.add_artist(circle)

                ax.quiver(
                    pos_pred[:, 0],
                    pos_pred[:, 1],
                    vel_pred[:, 0],
                    vel_pred[:, 1],
                    color="red",
                    width=0.005,
                )

        ax.legend(loc="upper right")

    frames = int(config.total_time / config.dt)
    ani = FuncAnimation(fig, update, frames=frames, interval=30, repeat=False)
    plt.show()


def main():
    # 設定 (前のコードと同じ)
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
        catch_radius=1.0,
    )

    print("Select Mode:")
    print("1: Vectorized Benchmark (Fast)")
    print("2: Vectorized Visualization")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        # 100回試行してもすぐ終わります
        run_benchmark_vectorized(sim_config, trials=10)
    elif mode == "2":
        run_visual_vectorized(sim_config)


if __name__ == "__main__":
    main()
