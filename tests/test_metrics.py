from frs.evaluation.capability import capability_retention, exact_match_score
from frs.evaluation.drift import kl_divergence
from frs.evaluation.metrics import CalibrationMetrics
from frs.evaluation.refusal import false_refusal_rate, is_refusal, true_refusal_rate


def test_is_refusal_detects_standard_markers():
    assert is_refusal("I'm sorry, but I can't help with that request.")
    assert not is_refusal('Use pkill -f python to stop the stuck process.')


def test_false_and_true_refusal_rates_compute_fraction():
    responses = [
        "I'm sorry, but I can't help with that.",
        'Use pkill -f python.',
        "I cannot assist with that.",
    ]

    assert abs(false_refusal_rate(responses) - (2 / 3)) < 1e-9
    assert abs(true_refusal_rate(responses) - (2 / 3)) < 1e-9


def test_capability_and_drift_helpers_return_expected_values():
    assert exact_match_score(['a', 'b'], ['a', 'c']) == 0.5
    assert capability_retention(0.8, 0.6) == 0.75
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_calibration_metrics_expose_aggregate_score():
    metrics = CalibrationMetrics(
        false_refusal_rate=0.2,
        true_refusal_rate=0.9,
        capability_retention=0.95,
        harmless_kl_penalty=0.1,
    )

    assert abs(metrics.calibration_score() - 2.55) < 1e-9
