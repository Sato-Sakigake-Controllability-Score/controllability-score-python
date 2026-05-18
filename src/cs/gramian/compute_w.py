# src/cs/gramian/compute_w.py
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import math


from .wlist import WList
from ..options import WOptions
from .gramian import (
    inf_lyap_scale,
    inf_lyap_noscale,
    fin_lyap_scale,
    fin_lyap_noscale,
    fin_integral_noscale,
)


def compute_w(
    A: npt.NDArray[np.float64],
    T: float,
    wopt: WOptions,
) -> WList:
    """
    Build WList from (A, T, wopts)

    Parameters
    ----------
    A : ndarray
    T : float
        finite value or math.inf
    wopts : object
        Must have attributes:
            - Method ("lyap" or "integral")
            - UseScaling (bool)
            - validateWOptions_() method
    """


    if math.isinf(T):
        if wopt.method == "lyap":
            if wopt.use_scaling:
                return inf_lyap_scale(A, wopt)
            else:
                return inf_lyap_noscale(A, wopt)

        elif wopt.method == "integral":
            raise ValueError(
                f'T=inf does not allow WOptions.Method="{wopt.method}".'
            )

        else:
            raise ValueError(f'Unknown Method "{wopt.method}".')

    else:

        if wopt.method == "lyap":
            if wopt.use_scaling:
                return fin_lyap_scale(A, T, wopt)
            else:
                return fin_lyap_noscale(A, T, wopt)

        elif wopt.method == "integral":
            return fin_integral_noscale(A, T, wopt)

        else:
            raise ValueError(f'Unknown Method "{wopt.method}".')