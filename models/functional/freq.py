import torch


def seq_fft(seq):
    f = torch.fft.rfft(seq, dim=1)
    amp = torch.absolute(f)
    phase = torch.angle(f)
    return amp, phase
