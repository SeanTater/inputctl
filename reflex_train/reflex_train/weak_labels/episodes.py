"""Episode segmentation for RL training.

Segments a recording session into episodes based on death/win events.
Each episode represents one "life" - from spawn to terminal event.
"""

from dataclasses import dataclass
from typing import Optional

from .events import Event


@dataclass
class Episode:
    """A single episode (life) within a recording session."""
    episode_id: int
    start_frame: int
    end_frame: int
    outcome: str  # "DEATH", "WIN", or "INCOMPLETE"
    reward: float  # -1 for death, +1 for win, 0 for incomplete
    length: int  # number of frames

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "outcome": self.outcome,
            "reward": self.reward,
            "length": self.length,
        }


def segment_episodes(
    events: list[Event],
    total_frames: int,
    respawn_gap: int = 30,
    death_reward: float = -1.0,
    win_reward: float = 1.0,
) -> list[Episode]:
    """Segment a session into episodes based on terminal events.

    Args:
        events: List of Event objects (deaths/wins) sorted by frame_idx
        total_frames: Total number of frames in the video
        respawn_gap: Frames to skip after death before new episode starts
        death_reward: Reward for death episodes
        win_reward: Reward for win episodes

    Returns:
        List of Episode objects
    """
    if total_frames <= 0:
        return []

    episodes = []
    current_start = 0
    episode_id = 0

    for event in events:
        if event.frame_idx <= current_start:
            # Event is before or at current episode start, skip
            continue

        # Create episode from current_start to this event
        if event.event == "DEATH":
            outcome = "DEATH"
            reward = death_reward
        elif event.event == "WIN":
            outcome = "WIN"
            reward = win_reward
        else:
            continue  # Unknown event type

        episode = Episode(
            episode_id=episode_id,
            start_frame=current_start,
            end_frame=event.frame_idx,
            outcome=outcome,
            reward=reward,
            length=event.frame_idx - current_start + 1,
        )
        episodes.append(episode)
        episode_id += 1

        # Start next episode after respawn gap
        current_start = event.frame_idx + respawn_gap

    # Handle remaining frames after last event
    if current_start < total_frames:
        episode = Episode(
            episode_id=episode_id,
            start_frame=current_start,
            end_frame=total_frames - 1,
            outcome="INCOMPLETE",
            reward=0.0,
            length=total_frames - current_start,
        )
        episodes.append(episode)

    return episodes


def compute_returns(
    episodes: list[Episode],
    events: list[Event] | None = None,
    gamma: float = 0.99,
    survival_bonus: float = 0.0,
    attack_reward: float = 0.1,
) -> dict[int, float]:
    """Compute discounted returns for each frame.

    Args:
        episodes: List of Episode objects
        events: List of all events (including ATTACK) for intermediate rewards
        gamma: Discount factor
        survival_bonus: Small per-frame reward for staying alive
        attack_reward: Bonus for killing/attacking enemies

    Returns:
        Dict mapping frame_idx -> return value
    """
    # Build attack frame set for fast lookup
    attack_frames = set()
    if events and attack_reward > 0:
        attack_frames = {e.frame_idx for e in events if e.event == "ATTACK"}

    returns = {}

    for ep in episodes:
        terminal_reward = ep.reward
        ep_length = ep.length

        # Collect attack rewards within this episode
        ep_attack_frames = [
            f for f in attack_frames
            if ep.start_frame <= f <= ep.end_frame
        ]

        for i, frame_idx in enumerate(range(ep.start_frame, ep.end_frame + 1)):
            frames_to_end = ep_length - i - 1

            # Return = discounted terminal reward
            discounted_terminal = (gamma ** frames_to_end) * terminal_reward

            # Add discounted attack rewards that occur after this frame
            attack_return = 0.0
            for attack_frame in ep_attack_frames:
                if attack_frame > frame_idx:
                    frames_to_attack = attack_frame - frame_idx
                    attack_return += (gamma ** frames_to_attack) * attack_reward

            # Survival bonus contribution (geometric sum)
            if survival_bonus > 0 and frames_to_end > 0:
                survival_return = survival_bonus * (1 - gamma ** frames_to_end) / (1 - gamma)
            else:
                survival_return = 0.0

            returns[frame_idx] = discounted_terminal + attack_return + survival_return

    return returns


def get_episode_for_frame(episodes: list[Episode], frame_idx: int) -> Optional[Episode]:
    """Find which episode a frame belongs to."""
    for ep in episodes:
        if ep.start_frame <= frame_idx <= ep.end_frame:
            return ep
    return None


def frame_is_near_death(
    frame_idx: int,
    episodes: list[Episode],
    window_frames: int = 30,
) -> bool:
    """Check if a frame is within window_frames of a death.

    Useful for downweighting frames immediately before death.
    """
    for ep in episodes:
        if ep.outcome == "DEATH":
            if ep.end_frame - window_frames <= frame_idx <= ep.end_frame:
                return True
    return False
