def overlap_add(frames, n_fft, hop_length, window, win_length):
    """
    istft(frame) -> overlap-add waveform
    :param frames: list of frames, like [frame1, frame2, ...]
    :param n_fft: number of fft points
    :param hop_length: hop size
    :param window: window type
    :param win_length: window length
    :return: overlap-added waveform
    """
    wavs = []
    for f in frames:
        wavs.append(torch.istft(f, n_fft=n_fft, hop_length=hop_length,
                                win_length=win_length, window=window, return_complex=False))
    # 对每一个wav，加上window对应的窗，例如hanning窗，就加上hanning窗
    wavs = torch.tensor([w*window for w in wavs])
    # 重叠相加
    y = wavs.sum(dim=0) / window.pow(2).sum(dim=0)
    return y