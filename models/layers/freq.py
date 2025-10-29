import math
import torch
import torch.nn as nn
import torch.fft as fft


# class BandedFourierLayer(nn.Module):
#     """
#         Fourier Block from CLF4SRec
#     """
#     def __init__(self, in_channels, out_channels, band, num_bands, length=201):
#         super().__init__()
#
#         self.length = length
#         self.total_freqs = (self.length // 2) + 1
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         self.band = band  # zero indexed
#         self.num_bands = num_bands
#
#         self.num_freqs = self.total_freqs // self.num_bands + (
#             self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)
#
#         self.start = self.band * (self.total_freqs // self.num_bands)
#         self.end = self.start + self.num_freqs
#
#         # case: from other frequencies
#         self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
#         self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
#         self.reset_parameters()
#
#     def forward(self, input):
#         # input - b t d
#         b, t, _ = input.shape
#         input_fft = fft.rfft(input, dim=1)
#         output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
#         output_fft[:, self.start:self.end] = self._forward(input_fft)
#         return fft.irfft(output_fft, n=input.size(1), dim=1)
#
#     def _forward(self, input):
#         output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
#         return output + self.bias
#
#     def reset_parameters(self) -> None:
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.bias, -bound, bound)


class BandedFourierLayer(nn.Module):
    """
    Fourier Block from CLF4SRec

    Issue:
    ----------------
    The original version of `BandedFourierLayer` defined its learnable parameters
    (`weight` and `bias`) as complex tensors (dtype=torch.cfloat). While this is
    mathematically convenient for Fourier-domain computations, PyTorch's built-in
    Adam optimizer does not support complex-valued parameters.

    During optimization, Adam uses grouped multi-tensor operations (e.g.,
    torch._foreach_add) that assume all parameters are real-valued tensors with
    compatible shapes. When complex parameters are included, their real and
    imaginary parts are internally represented as additional dimensions, causing
    shape mismatches such as:

        RuntimeError: The size of tensor a (2) must match the size of tensor b (64)
        at non-singleton dimension 3

    lower versions of pytorch might not support `foreach` function, this could be the
    reason why issue occurred here.

    Resolution:
    -----------
    In this revised implementation, all learnable parameters are stored as real-valued
    tensors (`weight_real`, `weight_imag`, `bias_real`, `bias_imag`). During the
    forward pass, these are combined into complex numbers using:

        torch.complex(real, imag)

    This preserves the same mathematical behavior as the original Fourier-based layer
    while keeping all parameters compatible with standard PyTorch optimizers (Adam,
    AdamW, SGD, etc.). Gradients are back-propagated separately through the real and
    imaginary components, ensuring stable training and correct gradient flow.

    """
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band  # zero indexed
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0)

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        # fixing: adam does not support cfloat anymore
        self.weight_real = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels)))
        self.weight_imag = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels)))
        self.bias_real = nn.Parameter(torch.empty((self.num_freqs, out_channels)))
        self.bias_imag = nn.Parameter(torch.empty((self.num_freqs, out_channels)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in [self.weight_real, self.weight_imag, self.bias_real, self.bias_imag]:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_real)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        for param in [self.bias_real, self.bias_imag]:
            nn.init.uniform_(param, -bound, bound)

    def _forward(self, input_fft):
        weight = torch.complex(self.weight_real, self.weight_imag)
        bias = torch.complex(self.bias_real, self.bias_imag)
        output = torch.einsum('bti,tio->bto', input_fft[:, self.start:self.end], weight)
        return output + bias

    def forward(self, input):
        # input: [B, T, D]
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)  # [B, T//2+1, D]
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels,
                                 device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)
