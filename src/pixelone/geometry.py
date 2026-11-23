"""Original-image coordinates and padding masks for the visual token grid."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class Layout:
    """Map row-major token and pixel indices to original screenshot coordinates."""

    width: int
    height: int
    stride: int = 32
    subdivisions: int = 32

    def __post_init__(self):
        if min(self.width, self.height) < 1 or self.stride != 32:
            raise ValueError('Positive image dimensions and the native 32-pixel stride required')
        if self.subdivisions < 1 or self.stride % self.subdivisions:
            raise ValueError('Subdivisions must divide the native stride')

    @property
    def cols(self):
        return math.ceil(self.width / self.stride)

    @property
    def rows(self):
        return math.ceil(self.height / self.stride)

    @property
    def count(self):
        return self.rows * self.cols

    @property
    def canvas_size(self):
        return self.cols * self.stride, self.rows * self.stride

    @property
    def fine_count(self):
        return self.subdivisions**2

    def positions(self, device=None):
        parent = torch.arange(self.count, device=device)
        fine = torch.arange(self.fine_count, device=device)
        unit = self.stride / self.subdivisions
        x = (parent[:, None] % self.cols) * self.stride + (
            fine[None] % self.subdivisions + 0.5
        ) * unit
        y = (parent[:, None] // self.cols) * self.stride + (
            fine[None] // self.subdivisions + 0.5
        ) * unit
        return torch.stack((x, y), -1)

    def valid(self, device=None):
        xy = self.positions(device)
        return (xy[..., 0] < self.width) & (xy[..., 1] < self.height)

    def center_target(self, box):
        if (
            len(box) != 4
            or not all(math.isfinite(v) for v in box)
            or not (0 <= box[0] < box[2] <= self.width and 0 <= box[1] < box[3] <= self.height)
        ):
            raise ValueError('Invalid original-pixel GT box')
        x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        col, row = int(x / self.stride), int(y / self.stride)
        unit = self.stride / self.subdivisions
        u, v = int((x - col * self.stride) / unit), int((y - row * self.stride) / unit)
        parent, fine = row * self.cols + col, v * self.subdivisions + u
        point = [(col * self.stride + (u + 0.5) * unit), (row * self.stride + (v + 0.5) * unit)]
        representable = box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
        return dict(
            parent=parent,
            fine=fine,
            center=[x, y],
            quantized_point=point,
            representable=representable,
        )


def inside(point, box):
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]
