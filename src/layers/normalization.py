from typing import List, Optional

import math
import numpy as np
from coremltools.converters.mil import Builder as mb
import coremltools.converters.mil as mil
from coremltools.converters.mil.mil import Operation, Var, types


def rmsnorm_anemll_style(x, axis: int, w: Optional[np.array] = None, prefix: str = ""):
    neg_x = mb.mul(x=x, y=np.array(-1, dtype=np.float16), name=prefix + "neg_x")
    x = mb.concat(values=(x, neg_x), axis=axis, name=prefix + "concat")
    x = mb.layer_norm(x=x, axes=(axis,), gamma=w, name=prefix + "layer_norm")
    x = mb.split(x=x, axis=axis, num_splits=2, name=prefix + "rms_norm")[0]
    return x
