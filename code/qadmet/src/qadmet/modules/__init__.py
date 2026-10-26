"""nn module package: backbones / fusion / heads / losses / multitask.

Add import-time registry registration modules along with Step 3 progress. Each sub-package
``__init__`` is responsible for its own registration trigger, so this ``__init__`` performs
re-export only (avoid heavy import).
"""

from . import losses  # noqa: F401  — register LOSSES / PER_TASK_LOSSES
from . import heads  # noqa: F401  — register HEADS
from . import fusion  # noqa: F401  — register FUSIONS
from . import backbones  # noqa: F401  — register BACKBONES (heavy deps lazy)

