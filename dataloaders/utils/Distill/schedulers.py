import torch 


class WeightDecayScheduler:
    def __init__(self, init_scale, total_steps, schedule_type="staged_linear"):
        self.init_scale = init_scale
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        
    def get_scale(self, step):
        if self.schedule_type == "staged_linear":
            start_step = 0.1 * self.total_steps
            end_step = 0.9 * self.total_steps
            if step < start_step:
                return self.init_scale
            elif step > end_step:
                return 0.0
            else:
                return self.init_scale * (1 - (step - start_step) / (end_step - start_step))
        elif self.schedule_type == "constant":
            return self.init_scale
        elif self.schedule_type == "sigmoid":
            return self.init_scale * torch.sigmoid(torch.tensor(step / self.total_steps * 14 - 4))
        else:
            raise ValueError(f"Invalid schedule type: {self.schedule_type}")

    def get_last_lr(self): 
        return [self.get_scale(self._step_count)]
    
    def step(self):
        self._step_count += 1
        scale = self.get_scale(self._step_count)
        return scale
    




class QuantScheduler:
    def __init__(self, total_steps, sigmoid_range=(-6, 8)):
        self.total_steps = total_steps
        self.sigmoid_range = sigmoid_range

    def get_scale(self, step): 
        scale = torch.sigmoid(
            (self.step / self.total_steps * (self.sigmoid_range[1] - self.sigmoid_range[0])
             + self.value_range[0])
        )
        return scale

    def get_last_lr(self):
        return [self.get_scale(self._step_count)]

    def step(self):
        self._step_count += 1
        scale = self.get_scale(self._step_count)
        return scale