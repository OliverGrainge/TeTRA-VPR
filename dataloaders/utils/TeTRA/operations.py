class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Apply the threshold operation: 1 if input > 0, else -1
        return torch.where(
            input > 0,
            torch.tensor(1.0, device=input.device),
            torch.tensor(-1.0, device=input.device),
        )

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient through unchanged (straight-through estimator)
        return grad_output
