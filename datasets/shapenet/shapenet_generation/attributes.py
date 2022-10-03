import itertools
import numpy as np
from typing import List, Tuple
from functools import cache


class Views:
    """
    Generates views based on inputs

    Args:
        view_start: float denoting starting orientation in degrees
        view_end: float denoting ending orientation in degrees
        num_views: number of views to render
    """

    def __init__(
        self,
        view_start=0.0,
        view_end=360.0,
        num_views=4,
        order="XYZ",
        canonical=(0, 0, 0),
    ):
        self.view_start = view_start
        self.view_end = view_end
        self.num_views = num_views
        self.order = order
        self.canonical = canonical

    @cache
    def generate(self) -> List[Tuple[int, int, int]]:
        return self.generate_planar() + self.generate_3d()

    def generate_planar(self) -> List[Tuple[int, int, int]]:
        views = [
            (0, int(angle), 0)
            for angle in np.linspace(
                self.view_start, self.view_end, self.num_views + 1
            )[:-1]
        ]
        return views

    def generate_3d(self) -> List[Tuple[int, int, int]]:
        views = [
            (int(angle), int(angle), int(angle))
            for angle in np.linspace(
                self.view_start, self.view_end, self.num_views + 1
            )[:-1]
        ]
        # remove (0, 0, 0) to avoid duplication from planar view
        views = views[1:]
        return views

    def __len__(self):
        return len(self.generate())

    def __repr__(self) -> str:
        return f"""start={self.view_start:0.1f}, end={self.view_end:0.1f}, 
                num_views={self.num_views}"""


class Backgrounds:
    pass
