import torch
from bidirectional_cross_attention import BidirectionalCrossAttention

video = torch.randn(1, 4096, 512)
audio = torch.randn(1, 8192, 386)

video_mask = torch.ones((1, 4096)).bool()
audio_mask = torch.ones((1, 8192)).bool()

joint_cross_attn = BidirectionalCrossAttention(
    dim = 512,
    heads = 8,
    dim_head = 64,
    context_dim = 386
)

video_out, audio_out = joint_cross_attn(
    video,
    audio,
    mask = video_mask,
    context_mask = audio_mask
)

assert video_out.shape == video.shape
assert audio_out.shape == audio.shape