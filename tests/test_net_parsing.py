"""
Unit tests for weConvert's .net file parsing helpers — these are used both
by the executable/libRR pipeline (existence checks, species counts) and,
now, by the bng propagator/sampler (which needs actual weighted values,
not just existence).
"""


class TestParseNetGroups:
    def test_weighted_terms_parsed_correctly(self, bare_weconvert, synthetic_net):
        """
        Regression test: groups used to keep only the first listed species
        and silently drop its coefficient. 'WeightedObs' is
        2*species2 + 3*species3 — both terms and both coefficients must
        survive parsing, in 0-based-index form.
        """
        wc = bare_weconvert()
        groups = wc._parse_net_groups(synthetic_net)
        assert groups["SimpleObs"] == [(0, 1.0)]
        assert groups["WeightedObs"] == [(1, 2.0), (2, 3.0)]

    def test_bare_index_defaults_to_coefficient_one(self, bare_weconvert, synthetic_net):
        wc = bare_weconvert()
        groups = wc._parse_net_groups(synthetic_net)
        idx, coeff = groups["SimpleObs"][0]
        assert idx == 0
        assert coeff == 1.0

    def test_unknown_observable_absent(self, bare_weconvert, synthetic_net):
        wc = bare_weconvert()
        groups = wc._parse_net_groups(synthetic_net)
        assert "NotAThing" not in groups


class TestParseNetSpecies:
    def test_species_count_and_order(self, bare_weconvert, synthetic_net):
        wc = bare_weconvert()
        species = wc._parse_net_species(synthetic_net)
        assert len(species) == 3
        # (1-based index, name, initial count)
        assert species[0] == (1, "A()", 20.0)
        assert species[1] == (2, "B()", 10.0)
        assert species[2] == (3, "C()", 5.0)
