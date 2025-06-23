import sympy as sm


def _is_symbolic(val):
    return issubclass(type(val), sm.Basic)  # pylint: disable=consider-using-ternary


def is_symbolic(val):
    if hasattr(val, "__iter__"):
        return any(_is_symbolic(s) for s in val)
    return _is_symbolic(val)
