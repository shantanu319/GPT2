"""The policy that decides what the model studies next.

One arm per data domain. Each round the loop trains a block on the arm the
director picks, then reports one reward: the mean drop in held-out probe loss
across *all* arms, not just the one that was trained. A domain that improves
itself by wrecking the others scores near zero, so not-forgetting is the thing
the director is maximising rather than a guard bolted on beside it.

Two details make exponential weights work on this signal:

Rank rescaling. Raw rewards shrink by orders of magnitude as the loss curve
flattens, which would freeze the update. A reward is replaced by its rank
inside a sliding window of recent rewards, mapped to [-1, 1], so the director
keeps responding to *relative* progress at any point on the curve.

Importance weighting. Only the chosen arm's reward is observed, so it is
divided by the probability of having observed it. Without it a domain that got
lucky early would compound its own head start into a collapse onto one arm.
Both are EXP3 as used for curricula in Graves et al. 2017, "Automated
Curriculum Learning for Neural Networks" (arXiv:1704.03003).
"""
import math
import random
from collections import deque


class Director:
    """Exponential-weights bandit over named arms.

    eta scales the log-weight step; the exploration floor bounds the
    importance weight at len(arms) / explore, so the largest single step a
    round can take is eta * len(arms) / explore. decay pulls the weights back
    toward uniform every round, which both bounds that random walk and keeps
    the director's memory finite so a stale preference cannot outlive the
    progress that earned it."""

    def __init__(self, arms, eta=0.08, explore=0.1, decay=0.01, window=32, seed=0):
        self.arms = list(arms)
        self.eta = eta
        self.explore = explore
        self.decay = decay
        self.log_w = [0.0] * len(self.arms)
        self.recent = deque(maxlen=window)
        self.rng = random.Random(seed)
        self.round = 0

    def probs(self):
        """Sampling distribution. Floored at explore/len(arms) per arm: an
        arm's value moves as the model moves, so none may be starved to zero
        and go stale."""
        top = max(self.log_w)
        weights = [math.exp(w - top) for w in self.log_w]
        total = sum(weights)
        floor = self.explore / len(self.arms)
        return [(1 - self.explore) * w / total + floor for w in weights]

    def choose(self):
        """Index of the arm to study this round."""
        probs = self.probs()
        draw = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if draw < acc:
                return i
        return len(probs) - 1

    def rescale(self, reward):
        """Reward's rank among the recent window, mapped to [-1, 1]. Returns 0
        on an empty window: the first reward carries no relative information."""
        if not self.recent:
            return 0.0
        below = sum(r < reward for r in self.recent)
        ties = sum(r == reward for r in self.recent)
        return 2.0 * (below + 0.5 * ties) / len(self.recent) - 1.0

    def update(self, arm_idx, reward):
        """Fold one round's raw reward into the weights; returns the rescaled
        reward actually used."""
        scaled = self.rescale(reward)
        self.recent.append(reward)
        self.log_w[arm_idx] += self.eta * scaled / self.probs()[arm_idx]
        # Weights only matter up to a shift, so re-centre, then pull toward
        # uniform: without that the log-weights are an unbiased random walk
        # and a run long enough will drift into a strong preference on reward
        # that was pure noise.
        mean = sum(self.log_w) / len(self.log_w)
        self.log_w = [(w - mean) * (1 - self.decay) for w in self.log_w]
        self.round += 1
        return scaled

    def state_dict(self):
        return {'arms': self.arms, 'log_w': self.log_w,
                'recent': list(self.recent), 'round': self.round,
                'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        if state['arms'] != self.arms:
            raise ValueError(f"arm set changed: {state['arms']} -> {self.arms}")
        self.log_w = list(state['log_w'])
        self.recent = deque(state['recent'], maxlen=self.recent.maxlen)
        self.round = state['round']
        self.rng.setstate(state['rng'])
