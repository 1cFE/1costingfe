"""Tests for radiation module."""

from costingfe.layers.radiation import compute_p_brem_rel


def test_p_brem_rel_reference_point():
    # n_e=1e20 m^-3, T_e=10 keV, Z_eff=1, V=1 m^3.
    # x=10/511=0.01957; sqrt(x)=0.1399; bracket=1.0515; 0.12113*0.1399*1.0515=0.01782 MW
    p = float(compute_p_brem_rel(1e20, 10.0, 1.0, 1.0))
    assert abs(p - 0.01782) < 4e-4


def test_p_brem_rel_matches_nonrel_at_low_T():
    # At 10 keV the relativistic+ee form is within ~6% of the non-rel Born value
    # 5.35e-3 * n20^2 * Z_eff * sqrt(T_e) * V = 5.35e-3*1*1*sqrt(10)*1 = 0.01692 MW
    p = float(compute_p_brem_rel(1e20, 10.0, 1.0, 1.0))
    assert 0.0169 <= p <= 0.0180
