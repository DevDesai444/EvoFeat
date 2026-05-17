"""Classical feature-engineering baselines.

Each baseline is exposed as a ``FeatureBuilder`` — a callable with the
signature ``(X_train, y_train, X_test) -> (X_train_new, X_test_new)``.
Same signature as the LLM-generated transforms; the eval harness in
``evofeat.evaluate`` doesn't know or care which kind it's holding.

The point of this module isn't to win on its own — it's to give a fair,
hyperparameter-light reference set for the headline comparison table.
Everything below uses sane defaults (no nested CV for the baseline's own
knobs); when we report numbers, we report them as 'classical method, with
its standard defaults' so the reader can replicate without reaching for
optuna.
"""

from evofeat.baselines.classical import (  # noqa: F401
    fisher_score_builder,
    mutual_info_builder,
    lasso_l1_builder,
    anova_f_builder,
    variance_threshold_builder,
    rfe_xgboost_builder,
    polynomial_builder,
    combined_classical_builder,
    BASELINES,
    BASELINES_REG,
    all_classification_baselines,
    all_regression_baselines,
)
