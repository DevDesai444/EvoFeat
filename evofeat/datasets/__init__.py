"""Dataset preparation helpers.

``banking77`` is the headline dataset for the cross-backbone comparison.
The legacy UCI CSVs under ``data/`` are still readable through
``evofeat.data.load_dataset`` for the auxiliary generalization sweep.
"""

from evofeat.datasets import banking77  # noqa: F401
