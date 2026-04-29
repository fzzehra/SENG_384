from typing import Tuple
 
Point = Tuple[float, float]
Triangle = Tuple[int, int, int]
 
 
class WarpingError(Exception):
    """Raised when warping cannot be completed safely."""
 
