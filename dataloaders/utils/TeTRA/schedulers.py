import torch 




class QuantScheduler:
    def __init__(self, total_steps, sigmoid_range=(-6, 8)):
        self.total_steps = total_steps
        self.sigmoid_range = sigmoid_range
        self._step_count = 0

    def get_scale(self, step):
        scale = torch.sigmoid(
            torch.tensor(
                (
                    self._step_count
                    / self.total_steps
                    * (self.sigmoid_range[1] - self.sigmoid_range[0])
                    + self.sigmoid_range[0]
                )
            )
        )
        return scale

    def get_last_lr(self):
        return [self.get_scale(self._step_count)]

    def step(self):
        self._step_count += 1
        scale = self.get_scale(self._step_count)
        return scale