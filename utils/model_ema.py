"""FP32 exponential moving averages without copying model forward closures."""

from contextlib import contextmanager
import math

import torch


class ModelEMA:
    def __init__(self, model, decay=0.9999):
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("EMA decay must be finite and in [0, 1)")
        self.decay = decay
        self.reset(model)

    @torch.no_grad()
    def reset(self, model):
        self.shadow = {
            name: value.detach().float().clone() if value.is_floating_point()
            else value.detach().clone()
            for name, value in model.state_dict().items()
        }
        self.num_updates = 0
        self.best_accuracy = float("-inf")
        self.best_depth = None

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            average = self.shadow[name]
            if value.is_floating_point():
                average.lerp_(value.detach().to(average), 1 - self.decay)
            else:
                average.copy_(value)
        self.num_updates += 1

    def state_dict(self):
        return dict(model_state=self.shadow, decay=self.decay,
                    num_updates=self.num_updates, best_accuracy=self.best_accuracy,
                    best_depth=self.best_depth)

    @torch.no_grad()
    def load_state_dict(self, state):
        if state['decay'] != self.decay:
            raise ValueError("Resume requires the same EMA decay")
        saved = state['model_state']
        if saved.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint keys do not match the model")
        for name, value in self.shadow.items():
            value.copy_(saved[name])
        self.num_updates = state['num_updates']
        self.best_accuracy = state['best_accuracy']
        self.best_depth = state.get('best_depth')

    @contextmanager
    def apply_to(self, model):
        """Restore training weights even if validation raises an exception."""
        state = model.state_dict()
        original = {name: value.detach().clone() for name, value in state.items()}
        try:
            with torch.no_grad():
                for name, value in state.items():
                    value.copy_(self.shadow[name])
            yield model
        finally:
            with torch.no_grad():
                for name, value in state.items():
                    value.copy_(original[name])


class MultiModelEMA(ModelEMA):
    """Independent averages of one training trajectory, with a legacy primary EMA."""

    def __init__(self, model, decays):
        decays = list(decays)
        if not decays or len(set(decays)) != len(decays):
            raise ValueError("EMA decays must be nonempty and distinct")
        super().__init__(model, decays[0])
        self.additional = [ModelEMA(model, decay) for decay in decays[1:]]

    @property
    def averages(self):
        return (self, *self.additional)

    def reset(self, model):
        super().reset(model)
        for average in getattr(self, 'additional', ()):
            average.reset(model)

    def update(self, model):
        super().update(model)
        for average in self.additional:
            average.update(model)

    def state_dict(self):
        state = super().state_dict()
        state['additional_ema_states'] = {
            str(average.decay): average.state_dict() for average in self.additional
        }
        return state

    def load_state_dict(self, state):
        saved = state.get('additional_ema_states', {})
        if set(saved) != {str(average.decay) for average in self.additional}:
            raise ValueError("Resume requires the same EMA decays and all EMA states")
        super().load_state_dict(state)
        for average in self.additional:
            average.load_state_dict(saved[str(average.decay)])
