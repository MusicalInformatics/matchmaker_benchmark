"""GaussianAudioHMM: an audio score follower using an HMM with chroma features.

Adapts matchmaker's GaussianAudioPitchHMM to use chroma features with a
cosine-similarity observation model (CosineExpAudioPitchObservationModel).
Uses matchmaker's HMM building blocks (observation model, gumbel_init_dist,
simple_transition_matrix) but runs the forward step efficiently by exploiting
the bidiagonal structure of the left-to-right transition matrix.
"""

import numpy as np

from matchmaker import register_method
from matchmaker.base import OnlineAlignment
from matchmaker.prob.hmm import (
    CosineExpAudioPitchObservationModel,
    gumbel_init_dist,
)
from matchmaker.utils.misc import generate_score_audio


class GaussianAudioHMMFollower(OnlineAlignment):
    """Frame-level audio score follower using matchmaker's cosine-exp
    observation model with an efficient windowed forward algorithm.

    The transition model is a left-to-right (stay/advance) HMM equivalent to
    matchmaker's simple_transition_matrix with stay_prob/trans_prob, but the
    forward step is computed in O(W) per frame (W = window size) instead of
    O(N^2) by exploiting the bidiagonal sparsity.
    """

    def __init__(
        self,
        reference_features,
        score_positions,
        queue=None,
        ref_frame_to_beat=None,
        rate=8.0,
        stay_prob=0.5,
        window_size=400,
        **kwargs,
    ):
        super().__init__(
            score_positions=score_positions,
            queue=queue,
            reference_features=reference_features,
        )
        self.ref_frame_to_beat = ref_frame_to_beat
        self.n_ref_states = len(reference_features)
        self.window_size = window_size
        self.stay_prob = stay_prob
        self.advance_prob = 1.0 - stay_prob

        self.observation_model = CosineExpAudioPitchObservationModel(
            audio_features=reference_features,
            rate=rate,
        )
        self._alpha = gumbel_init_dist(n_states=self.n_ref_states).astype(
            np.float64
        )
        self._alpha /= self._alpha.sum()

    def step(self, features) -> None:
        best = self.current_index
        lo = max(0, best - 20)
        hi = min(self.n_ref_states, best + self.window_size)

        alpha_win = self._alpha[lo:hi].copy()

        new_alpha = np.empty_like(alpha_win)
        new_alpha[0] = self.stay_prob * alpha_win[0]
        new_alpha[1:] = (
            self.stay_prob * alpha_win[1:]
            + self.advance_prob * alpha_win[:-1]
        )

        obs_prob = self.observation_model(features)
        new_alpha *= obs_prob[lo:hi]

        total = new_alpha.sum()
        if total > 0:
            new_alpha /= total

        self._alpha[lo:hi] = new_alpha
        mass_outside = self._alpha[:lo].sum() + self._alpha[hi:].sum()
        if mass_outside > 0:
            self._alpha[:lo] = 0.0
            self._alpha[hi:] = 0.0
            renorm = self._alpha.sum()
            if renorm > 0:
                self._alpha /= renorm

        current_state = lo + int(np.argmax(new_alpha))
        self.current_index = current_state
        self.observation_model.current_state = current_state

    def is_still_following(self) -> bool:
        return self.current_index < self.n_ref_states - 1

    def get_current_position(self) -> float:
        if self.ref_frame_to_beat is not None and self.current_index < len(
            self.ref_frame_to_beat
        ):
            return float(self.ref_frame_to_beat[self.current_index])
        return float(self.score_positions[self.current_index])


def build_reference(mm):
    """Synthesise the score to audio and extract chroma features."""
    audio = generate_score_audio(mm.score_part, mm.tempo, mm.sample_rate)
    features, _ = mm.processor((audio.astype(np.float32), 0.0))
    mm.processor.reset()
    return features


def build_follower(mm):
    return GaussianAudioHMMFollower(
        reference_features=mm.reference_features,
        score_positions=mm.score_positions,
        queue=mm.stream.queue,
        ref_frame_to_beat=mm.ref_frame_to_beat(),
        rate=8.0,
        stay_prob=0.5,
        window_size=400,
    )


register_method(
    "gaussian-audio-hmm",
    input_type="audio",
    build_follower=build_follower,
    build_reference=build_reference,
    default_kwargs={
        "processor": "chroma",
        "sample_rate": 44100,
        "frame_rate": 30,
    },
)
