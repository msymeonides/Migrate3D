import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Callable
from dataclasses import dataclass


@dataclass
class TrackConfig:
    """Configuration for track generation."""

    # =============================================================================
    # OUTPUT FILES
    # =============================================================================
    output_tracks_file: str = 'synthetic-3D_segments.csv'
    output_categories_file: str = 'synthetic-3D_categories.csv'

    # =============================================================================
    # GENERATION PARAMETERS
    # =============================================================================
    n_tracks: int = 50  # Number of tracks to generate per category

    # =============================================================================
    # 2D/3D CONFIGURATION
    # =============================================================================
    generate_3d: bool = True                            # Enable 3D coordinates
    z_velocity_range: Tuple[float, float] = (0.1, 2.0)  # Z movement speed
    z_jitter_range: Tuple[float, float] = (0.05, 0.5)   # Z velocity noise

    # Z-axis scaling factors (only used when generate_3d=True)
    z_movement_scale: float = 2.0  # Multiplier for all Z movements (higher = more vertical motion)
    z_turn_scale: float = 2.0      # Multiplier for Z direction changes (higher = more 3D turning)
    z_burst_scale: float = 2.0     # Multiplier for Z burst movements (higher = more vertical bursts)

    # =============================================================================
    # STANDARDIZATION CONTROLS - Override natural track characteristics
    # =============================================================================
    standardize_duration: bool = True
    standardize_path_length: bool = True
    target_duration: Optional[int] = 200          # Target track duration (time steps)
    target_path_length: Optional[float] = 500.0   # Target total distance traveled
    duration_variation: float = 0.1               # ±10% variation around target_duration
    path_length_variation: float = 0.1            # ±10% variation around target_path_length

    # =============================================================================
    # NATURAL TRACK GENERATION - Used when standardization is disabled
    # =============================================================================
    velocity_range: Tuple[float, float] = (8, 14)     # Base movement speed (units/time step)
    jitter_range: Tuple[float, float] = (0.2, 3.0)    # Random velocity noise (higher = more erratic)
    distance_range: Tuple[float, float] = (150, 300)  # Overall track scale
    # NOTE: distance_range only affects initial duration calculation when standardize_duration=False
    # NOTE: When standardize_path_length=True, final path length ignores distance_range

    # =============================================================================
    # PARAMETER INTERACTION SUMMARY:
    #
    # When standardize_duration=True:
    #   - target_duration & duration_variation control final track length
    #   - distance_range becomes less relevant (only affects initial calculation)
    #   - velocity_range & jitter_range still affect movement characteristics
    #
    # When standardize_path_length=True:
    #   - target_path_length & path_length_variation control final distance traveled
    #   - distance_range becomes irrelevant for final path length
    #   - velocity_range & jitter_range still affect movement speed patterns
    #
    # When generate_3d=False:
    #   - All z_* parameters are ignored
    #   - Only X,Y coordinates are generated
    #
    # When generate_3d=True:
    #   - z_velocity_range & z_jitter_range control Z movement patterns
    #   - z_*_scale parameters control intensity of 3D effects
    # =============================================================================

class SyntheticTrackFactory:
    """Factory for generating synthetic tracks with configurable behavior."""

    def __init__(self, config: Optional[TrackConfig] = None):
        self.config = config or TrackConfig()
        self.generators = self._get_default_generators()

    def _get_default_generators(self) -> Dict[str, Callable]:
        """Return default track generation functions."""
        return {
            'MostlyStraight': lambda cfg: self._generate_mostly_straight_track(cfg),
            'Turns45deg': lambda cfg: self._generate_turn_track(cfg, (35, 55), (35, 45), (8, 12)),
            'Turns85deg': lambda cfg: self._generate_turn_track(cfg, (75, 95), (35, 45), (8, 12)),
            'TurnsRandom': lambda cfg: self._generate_turn_track(cfg, (10, 170), (35, 45), (8, 12)),
            'Burst': lambda cfg: self._generate_burst_track(cfg, (0.2, 0.3)),
            'Back-forth': lambda cfg: self._generate_back_forth_track(cfg, (15, 25), (1, 3)),
            'Dead': lambda cfg: self._generate_back_forth_track(cfg, (0.5, 3.0), (1, 3))
        }

    def generate_tracks(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete track dataset."""
        np.random.seed(42)
        all_tracks = []
        all_categories = []

        for i in range(self.config.n_tracks):
            for category_name, generator in self.generators.items():
                # Create individual track config
                track_cfg = self._create_track_config(category_name, i)

                # Generate track
                track_data = generator(track_cfg)
                track_df = self._create_track_dataframe(track_cfg['object_id'], track_data)

                all_tracks.append(track_df)
                all_categories.append({
                    'Object ID': track_cfg['object_id'],
                    'Category': category_name
                })

        df_tracks = pd.concat(all_tracks, ignore_index=True)
        df_categories = pd.DataFrame(all_categories)

        return df_tracks, df_categories

    def save_tracks(self, df_tracks: pd.DataFrame, df_categories: pd.DataFrame):
        """Save tracks to configured files."""
        df_tracks.to_csv(self.config.output_tracks_file, index=False)
        df_categories.to_csv(self.config.output_categories_file, index=False)

    def _create_track_config(self, category_name: str, index: int) -> Dict:
        """Create configuration for individual track."""
        velocity = np.random.uniform(*self.config.velocity_range)
        jitter = np.random.uniform(*self.config.jitter_range)
        start_angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(*self.config.distance_range)

        # Calculate base total_points
        total_points = int(distance / velocity)

        # Apply duration variation if standardization is enabled
        if self.config.standardize_duration and self.config.target_duration:
            # Add random variation around target duration
            variation = np.random.uniform(-self.config.duration_variation, self.config.duration_variation)
            total_points = int(self.config.target_duration * (1 + variation))
            total_points = max(total_points, 10)  # Ensure minimum duration

        config = {
            'object_id': hash(category_name) % 10000 * 1000 + index,
            'start_pos': (0, 0) if not self.config.generate_3d else (0, 0, 0),
            'velocity': velocity,
            'velocity_jitter': jitter,
            'start_angle': start_angle,
            'total_points': total_points,
            'is_3d': self.config.generate_3d
        }

        # 3D-specific parameters
        if self.config.generate_3d:
            config['z_velocity'] = np.random.uniform(*self.config.z_velocity_range)
            config['z_jitter'] = np.random.uniform(*self.config.z_jitter_range)
            config['z_angle'] = np.random.uniform(0, 2 * np.pi)

        return config

    def _create_track_dataframe(self, object_id: int, track_data: Dict) -> pd.DataFrame:
        """Create standardized DataFrame from track data."""
        positions, times = track_data['positions'], track_data['times']

        # Apply global constraints with variation
        if self.config.standardize_duration and self.config.target_duration:
            # Duration is already handled in _create_track_config, but we still need to resample
            # if the generated track doesn't match the varied target
            target_duration = len(times)  # Use the already varied duration
            if len(positions) != target_duration:
                positions, times = self._standardize_duration_to_target(positions, times, target_duration)

        # Check if this is a dead track by looking at the object_id pattern
        # Dead tracks have object_id based on hash('Dead') % 10000 * 1000 + index
        is_dead_track = (object_id // 1000) == (hash('Dead') % 10000)

        # Apply path length standardization only to non-dead tracks
        if (self.config.standardize_path_length and
            self.config.target_path_length and
            not is_dead_track):
            positions = self._standardize_path_length_with_variation(positions)

        # Handle 2D vs 3D coordinates
        if self.config.generate_3d:
            return pd.DataFrame({
                'Object ID': object_id,
                'Time': times,
                'X Coordinate': [pos[0] for pos in positions],
                'Y Coordinate': [pos[1] for pos in positions],
                'Z Coordinate': [pos[2] for pos in positions]
            })
        else:
            return pd.DataFrame({
                'Object ID': object_id,
                'Time': times,
                'X Coordinate': [pos[0] for pos in positions],
                'Y Coordinate': [pos[1] for pos in positions],
                'Z Coordinate': [0] * len(positions)
            })

    def _standardize_duration_to_target(self, positions: List[Tuple], times: List[int], target_duration: int) -> Tuple[List[Tuple], List[int]]:
        """Resample track to specific target duration."""
        if len(positions) == target_duration:
            return positions, times

        original_indices = np.linspace(0, len(positions) - 1, len(positions))
        target_indices = np.linspace(0, len(positions) - 1, target_duration)

        pos_array = np.array(positions)
        new_x = np.interp(target_indices, original_indices, pos_array[:, 0])
        new_y = np.interp(target_indices, original_indices, pos_array[:, 1])

        if self.config.generate_3d:
            new_z = np.interp(target_indices, original_indices, pos_array[:, 2])
            return [(x, y, z) for x, y, z in zip(new_x, new_y, new_z)], list(range(target_duration))
        else:
            return [(x, y) for x, y in zip(new_x, new_y)], list(range(target_duration))

    def _standardize_path_length_with_variation(self, positions: List[Tuple]) -> List[Tuple]:
        """Scale track to target path length with variation."""
        if len(positions) < 2:
            return positions

        current_length = sum(np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i]))
                           for i in range(len(positions) - 1))

        if current_length == 0:
            return positions

        # Apply variation to target path length
        variation = np.random.uniform(-self.config.path_length_variation, self.config.path_length_variation)
        target_length = self.config.target_path_length * (1 + variation)

        scale = target_length / current_length
        start_pos = np.array(positions[0])

        return [positions[0]] + [tuple(start_pos + (np.array(pos) - start_pos) * scale)
                                for pos in positions[1:]]

    def _generate_mostly_straight_track(self, cfg: Dict) -> Dict:
        """Generate mostly straight track with occasional slight turns."""
        np.random.seed(cfg['object_id'])

        # More frequent but still infrequent turns (every 50-80 steps instead of 80-120)
        turn_interval = np.random.randint(50, 80)
        n_turns = max(2, cfg['total_points'] // turn_interval)  # Ensure at least 2 turns

        # Generate turn positions
        turn_starts = []
        for i in range(n_turns):
            turn_start = min(i * turn_interval + np.random.randint(-10, 10), cfg['total_points'] - 15)
            if turn_start > 10:  # Avoid turns too early
                turn_starts.append(turn_start)

        positions = [cfg['start_pos']]
        times = [0]
        direction = cfg['start_angle']

        # Lower velocity variation for steadier movement
        base_velocity = cfg['velocity']
        velocity_noise = cfg['velocity_jitter'] * 0.3  # Reduce jitter for steadier movement

        if cfg['is_3d']:
            z_direction = cfg['z_angle']
            z_velocities = np.random.normal(
                cfg['z_velocity'] * self.config.z_movement_scale * 0.5,  # Less Z movement
                cfg['z_jitter'] * 0.3,
                cfg['total_points']
            )

        turn_idx = 0
        in_turn = False
        turn_end = 0

        for t in range(1, cfg['total_points']):
            # Check if starting a turn
            if turn_idx < len(turn_starts) and t >= turn_starts[turn_idx] and not in_turn:
                in_turn = True
                turn_end = t + np.random.randint(12, 20)  # Longer turn duration (12-20 steps)

            # Execute turn
            if in_turn and t <= turn_end:
                # More noticeable turn (10-25 degrees instead of 5-15)
                turn_angle = np.random.uniform(10, 25)
                if np.random.random() < 0.5:
                    turn_angle = -turn_angle
                # Apply turn more gradually but with more total change
                direction += np.deg2rad(turn_angle / (turn_end - turn_starts[turn_idx] + 1))

                if cfg['is_3d']:
                    z_turn = np.random.uniform(-5, 5) * self.config.z_turn_scale
                    z_direction += np.deg2rad(z_turn / (turn_end - turn_starts[turn_idx] + 1))

            # End turn
            if in_turn and t > turn_end:
                in_turn = False
                turn_idx += 1

            # Calculate movement with slight random variation even when not turning
            velocity = np.random.normal(base_velocity, velocity_noise)

            # Add very slight random direction noise even during straight segments
            if not in_turn:
                direction += np.random.normal(0, 0.02)  # Very small random drift

            dx = velocity * np.cos(direction)
            dy = velocity * np.sin(direction)

            if cfg['is_3d']:
                dz = z_velocities[t] * np.cos(z_direction)
                new_pos = (positions[-1][0] + dx, positions[-1][1] + dy, positions[-1][2] + dz)
            else:
                new_pos = (positions[-1][0] + dx, positions[-1][1] + dy)

            positions.append(new_pos)
            times.append(t)

        return {'positions': positions, 'times': times}

    def _generate_turn_track(self, cfg: Dict, turn_angle_range: Tuple[float, float],
                            turn_interval_range: Tuple[int, int], turn_duration_range: Tuple[int, int]) -> Dict:
        """Generate track with regular turns."""
        np.random.seed(cfg['object_id'])

        # Calculate turn timing
        turn_interval = np.random.randint(*turn_interval_range)
        turn_duration = np.random.randint(*turn_duration_range)

        # Generate turn positions
        turn_starts = []
        t = turn_interval
        while t < cfg['total_points'] - turn_duration:
            turn_starts.append(t)
            t += turn_interval + np.random.randint(-5, 5)  # Slight variation

        positions = [cfg['start_pos']]
        times = [0]
        direction = cfg['start_angle']
        velocities = np.random.normal(cfg['velocity'], cfg['velocity_jitter'], cfg['total_points'])

        if cfg['is_3d']:
            z_direction = cfg['z_angle']
            z_velocities = np.random.normal(
                cfg['z_velocity'] * self.config.z_movement_scale,
                cfg['z_jitter'] * self.config.z_movement_scale,
                cfg['total_points']
            )

        turn_idx = 0
        in_turn = False
        turn_end = 0

        for t in range(1, cfg['total_points']):
            # Check if starting a turn
            if turn_idx < len(turn_starts) and t >= turn_starts[turn_idx] and not in_turn:
                in_turn = True
                turn_end = t + turn_duration

            # Execute turn
            if in_turn and t <= turn_end:
                # Gradual turn over the duration
                total_turn = np.random.uniform(*turn_angle_range)
                if np.random.random() < 0.5:
                    total_turn = -total_turn
                turn_increment = total_turn / turn_duration
                direction += np.deg2rad(turn_increment)

                if cfg['is_3d']:
                    z_turn = np.random.uniform(-10, 10) * self.config.z_turn_scale / turn_duration
                    z_direction += np.deg2rad(z_turn)

            # End turn
            if in_turn and t > turn_end:
                in_turn = False
                turn_idx += 1

            # Calculate movement
            dx = velocities[t] * np.cos(direction)
            dy = velocities[t] * np.sin(direction)

            if cfg['is_3d']:
                dz = z_velocities[t] * np.cos(z_direction)
                new_pos = (positions[-1][0] + dx, positions[-1][1] + dy, positions[-1][2] + dz)
            else:
                new_pos = (positions[-1][0] + dx, positions[-1][1] + dy)

            positions.append(new_pos)
            times.append(t)

        return {'positions': positions, 'times': times}

    def _generate_burst_track(self, cfg: Dict, idle_speed_range: Tuple[float, float]) -> Dict:
        """Generate burst track with brief period of high activity."""
        np.random.seed(cfg['object_id'])

        # Longer burst parameters
        burst_duration = np.random.randint(25, 45)  # Longer burst (25-45 steps instead of 15-30)
        burst_start = np.random.randint(int(cfg['total_points'] * 0.2), int(cfg['total_points'] * 0.8))

        idle_speed = np.random.uniform(*idle_speed_range) * cfg['velocity']  # 20-30% of normal
        burst_speed = 1.75 * cfg['velocity']  # 175% of normal

        positions = [cfg['start_pos']]
        times = [0]
        pos = np.array(cfg['start_pos'], dtype=float)
        direction = cfg['start_angle']

        if cfg['is_3d']:
            z_direction = cfg['z_angle']

        # Turn parameters - execute over ~10 steps
        turn_start = burst_start + burst_duration // 3  # Start turn in first third of burst
        turn_duration = 10
        turn_end = turn_start + turn_duration
        total_turn_angle = np.random.uniform(35, 55)  # Total turn amount
        if np.random.random() < 0.5:
            total_turn_angle = -total_turn_angle
        turn_per_step = total_turn_angle / turn_duration

        if cfg['is_3d']:
            total_z_turn = np.random.uniform(-15, 15) * self.config.z_turn_scale
            z_turn_per_step = total_z_turn / turn_duration

        for t in range(1, cfg['total_points']):
            is_burst = burst_start <= t < burst_start + burst_duration
            is_turning = turn_start <= t < turn_end

            if is_burst:
                # High speed during burst
                velocity = np.random.normal(burst_speed, cfg['velocity_jitter'])

                # Execute gradual turn over ~10 steps
                if is_turning:
                    direction += np.deg2rad(turn_per_step)
                    if cfg['is_3d']:
                        z_direction += np.deg2rad(z_turn_per_step)

                if cfg['is_3d']:
                    z_velocity = np.random.normal(
                        cfg['z_velocity'] * self.config.z_burst_scale,
                        cfg['z_jitter']
                    )
            else:
                # Slow, steady movement
                velocity = np.random.normal(idle_speed, cfg['velocity_jitter'] * 0.3)

                if cfg['is_3d']:
                    z_velocity = np.random.normal(
                        cfg['z_velocity'] * 0.3,
                        cfg['z_jitter'] * 0.3
                    )

            # Calculate movement
            dx = velocity * np.cos(direction)
            dy = velocity * np.sin(direction)

            if cfg['is_3d']:
                dz = z_velocity * np.cos(z_direction)
                new_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
            else:
                new_pos = (pos[0] + dx, pos[1] + dy)

            pos = np.array(new_pos)
            positions.append(tuple(pos))
            times.append(t)

        return {'positions': positions, 'times': times}

    def _generate_back_forth_track(self, cfg: Dict, distance_range: Tuple[float, float],
                                  pause_range: Tuple[int, int]) -> Dict:
        """Generate back-and-forth track between two points (uses distance_range[0] as the separation)."""
        np.random.seed(cfg['object_id'])

        # Use the first value of distance_range as the separation distance
        # For back-forth: distance_range = (8, 12) -> use ~10 units
        # For dead: distance_range = (0.1, 1.0) -> use ~0.5 units
        travel_distance = distance_range[0] * np.random.uniform(0.8, 1.2)

        # Define the two points
        angle = np.random.uniform(0, 2 * np.pi)
        point_a = np.array(cfg['start_pos'], dtype=float)

        if cfg['is_3d']:
            z_angle = np.random.uniform(0, 2 * np.pi)
            point_b = point_a + travel_distance * np.array([
                np.cos(angle),
                np.sin(angle),
                0.3 * np.cos(z_angle)
            ])
        else:
            point_b = point_a + travel_distance * np.array([np.cos(angle), np.sin(angle)])

        positions = [tuple(point_a)]
        times = [0]
        current_pos = point_a.copy()

        # Movement parameters - scale with distance
        going_to_b = True
        movement_speed = cfg['velocity'] * min(0.8, travel_distance / 10.0)  # Slower for smaller distances
        pause_steps_remaining = 0

        for t in range(1, cfg['total_points']):
            # Determine current target
            target_base = point_b if going_to_b else point_a

            # Add noise proportional to travel distance
            target_noise = np.random.normal(0, travel_distance * 0.1, len(target_base))
            target = target_base + target_noise

            # Calculate direction to target
            direction_vec = target - current_pos
            distance_to_target = np.linalg.norm(direction_vec)

            # Scale pause threshold with travel distance
            pause_threshold = max(0.5, travel_distance * 0.2)

            # If we're close to target, start/continue pause
            if distance_to_target < pause_threshold:
                if pause_steps_remaining <= 0:
                    # Start new pause
                    pause_steps_remaining = np.random.randint(*pause_range)
                    # Switch direction after pause
                    going_to_b = not going_to_b

                # During pause, small random movements scaled by travel distance
                if pause_steps_remaining > 0:
                    noise_scale = max(0.05, travel_distance * 0.02)
                    noise = np.random.normal(0, noise_scale, len(current_pos))
                    current_pos += noise
                    pause_steps_remaining -= 1

            else:
                # Move toward target
                if np.linalg.norm(direction_vec) > 0:
                    direction_vec = direction_vec / np.linalg.norm(direction_vec)

                # Add perpendicular wandering scaled by travel distance
                if len(current_pos) == 2:
                    perp_vec = np.array([-direction_vec[1], direction_vec[0]])
                else:
                    perp_vec = np.random.normal(0, 1, 3)
                    perp_vec = perp_vec - np.dot(perp_vec, direction_vec) * direction_vec
                    if np.linalg.norm(perp_vec) > 0:
                        perp_vec = perp_vec / np.linalg.norm(perp_vec)

                # Scale movement and wandering with travel distance
                wander_amount = np.random.normal(0, travel_distance * 0.05)
                step_size = np.random.normal(movement_speed, cfg['velocity_jitter'] * 0.2)
                step_size = max(0.01, step_size)  # Minimum step size

                movement = step_size * direction_vec + wander_amount * perp_vec
                current_pos += movement

            positions.append(tuple(current_pos))
            times.append(t)

        return {'positions': positions, 'times': times}

if __name__ == "__main__":
    # Standard usage
    factory = SyntheticTrackFactory()
    df_tracks, df_categories = factory.generate_tracks()
    factory.save_tracks(df_tracks, df_categories)
