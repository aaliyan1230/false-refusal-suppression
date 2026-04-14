from frs.editing.apply_edit import EditSpec, apply_directional_ablation
from frs.editing.directions import l2_norm



def test_apply_directional_ablation_removes_projected_component():
    matrix = [[2.0, 0.0], [0.0, 3.0]]
    direction = [1.0, 0.0]

    edited = apply_directional_ablation(matrix, direction, EditSpec(strength=1.0, norm_preserving=False))

    assert edited[0][0] == 0.0
    assert edited[1][1] == 3.0



def test_norm_preserving_edit_keeps_row_norms():
    matrix = [[3.0, 4.0], [4.0, 3.0]]
    direction = [1.0, 0.0]

    edited = apply_directional_ablation(matrix, direction, EditSpec(strength=1.0, norm_preserving=True))

    assert abs(l2_norm(matrix[0]) - l2_norm(edited[0])) < 1e-9
    assert abs(l2_norm(matrix[1]) - l2_norm(edited[1])) < 1e-9
