from frs.editing.directions import direction_from_contrast, l2_norm, separability_score
from frs.editing.projection import orthogonalize


def test_direction_from_contrast_returns_unit_vector():
    direction = direction_from_contrast([[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]])

    assert len(direction) == 2
    assert abs(l2_norm(direction) - 1.0) < 1e-9


def test_orthogonalize_removes_reference_component():
    candidate = [1.0, 1.0]
    reference = [1.0, 0.0]

    orthogonal = orthogonalize(candidate, reference)

    assert abs(orthogonal[0]) < 1e-9
    assert abs(orthogonal[1] - 1.0) < 1e-9


def test_separability_score_is_positive_for_distinct_groups():
    score = separability_score([[2.0, 2.0], [2.0, 2.0]], [[0.0, 0.0], [0.0, 0.0]])

    assert score > 0
