import torch


class QuantScheduler:
    def __init__(self, total_steps, scheduler_type="sigmoid", sigmoid_range=(-6, 8)):
        self.total_steps = total_steps
        self.sigmoid_range = sigmoid_range
        self.scheduler_type = scheduler_type
        self._step_count = 0

    def get_scale(self, step):
        if self.scheduler_type == "sigmoid":
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
        elif self.scheduler_type == "constant":
            scale = 0.2
        elif self.scheduler_type == "linear":
            scale = self._step_count / self.total_steps
        else:
            raise NotImplementedError(
                f"Scheduler type {self.scheduler_type} not implemented"
            )
        return scale

    def get_last_lr(self):
        return [self.get_scale(self._step_count)]

    def step(self):
        self._step_count += 1
        scale = self.get_scale(self._step_count)
        return scale