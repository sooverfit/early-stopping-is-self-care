"""early-stopping-is-self-care: a joke early-stopping companion callback.

The core has zero dependencies. Framework callbacks live in submodules:

    from early_stopping_is_self_care.keras import SelfCareCallback
    from early_stopping_is_self_care.lightning import SelfCareCallback
"""

from .core import Companion, PlateauTracker, compose_roast, roast
from .diagnose import Diagnosis, diagnose

__version__ = "0.1.0"
__all__ = ["Companion", "Diagnosis", "PlateauTracker", "compose_roast", "diagnose", "roast",
           "__version__"]
