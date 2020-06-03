import torch


# class LinearQuantizer(torch.autograd.Function):
#     """
#     Linear quantizer according to
#     Miyashita et al., Convolutional Neural Networks using Logarithmic Data Representation.
#     For symmetric limits (min_x, max_x) around zero, this quantizer also has an output 0. However, the largest value
#     is not attained by this method.
#     Example: num_bits=2, min_x=-1, max_x=1 => {-1, -0.5, 0, 0.5}
#     """
#
#     @staticmethod
#     def forward(ctx, input, num_bits, min_x, max_x):
#         num_vals = 2.0 ** num_bits
#         step = 2.0 ** -num_bits
#         output = (input - min_x) * (num_vals / (max_x - min_x))  # transform to [0,num_vals]
#         output = torch.round(output)
#         output = torch.clamp(output, 0, num_vals - 1.0)  # clip values
#         output = output * (step * (max_x - min_x)) + min_x
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None, None, None


class LinearQuantizerDorefa(torch.autograd.Function):
    """
    Linear quantizer according to
    Zhou et al., DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients, arXiv:1606.06160 (2016)
    This quantizer quantized the weights evently between min_x and max_x. This function is called quantize_k in the
    paper.
    Example: num_bits=2, min_x=-1, max_x=1 => {-1, -1/3, 1/3, 1}
    """

    @staticmethod
    def forward(ctx, input, num_bits, min_x, max_x):
        output = (input - min_x) * ((2.0 ** num_bits - 1.0) / (max_x - min_x))  # transform to [0,2**num_bits-1]
        output = torch.round(output) * (1.0 / (2.0 ** num_bits - 1.0))
        output = torch.clamp(output, 0, 1)
        output = output * (max_x - min_x) + min_x
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
