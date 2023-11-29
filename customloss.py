import torch

class MyCumulativeLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, previous_loss, predicted, target):
        # Compute the loss for the current batch
        loss = torch.mean((predicted - target) ** 2)
        
        # Add the current loss to the previous_loss
        cumulative_loss = previous_loss + loss

        # Save cumulative_loss for the backward pass
        ctx.save_for_backward(predicted, target)

        return cumulative_loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved tensors
        predicted, target = ctx.saved_tensors

        # Compute the gradient of the loss with respect to predicted
        grad_predicted = 2 * (predicted - target) * grad_output

        # Set previous_loss to None as it doesn't have gradients
        grad_previous_loss = None

        return grad_previous_loss, grad_predicted, None

# Example usage:
previous_loss = torch.tensor([0.0], requires_grad=True)
predicted = torch.tensor([2.0], requires_grad=True)
target = torch.tensor([3.0])

loss_fn = MyCumulativeLoss.apply
cumulative_loss1 = loss_fn(previous_loss, predicted, target)
# cumulative_loss2 = loss_fn(cumulative_loss1, predicted, target)
# cumulative_loss3 = loss_fn(cumulative_loss2, predicted, target)

# Backpropagation
cumulative_loss1.backward()

# The gradient for previous_loss will be None as it's a constant
# The gradient for predicted will be computed based on the accumulated loss
print(previous_loss.grad)  # None
print(predicted.grad)  # [4.0]
