import pytest
from dms_quant_framework.plotting import colors_for_sequence


class TestColorsForSequence:
    def test_valid_input(self):
        # Test with valid DNA sequence
        assert colors_for_sequence("ATCG") == ["red", "green", "blue", "orange"]

        # Test with valid RNA sequence
        assert colors_for_sequence("AUCG") == ["red", "green", "blue", "orange"]

        # Test with mixed case
        assert colors_for_sequence("aTcG") == ["red", "green", "blue", "orange"]

        # Test with '&' character
        assert colors_for_sequence("A&CG") == ["red", "gray", "blue", "orange"]

    def test_empty_input(self):
        assert colors_for_sequence("") == []

    def test_invalid_character(self):
        with pytest.raises(ValueError) as exc_info:
            colors_for_sequence("ATCGX")
        assert "Invalid character 'X' in sequence" in str(exc_info.value)

    def test_all_characters(self):
        sequence = "ACGTU&"
        expected = ["red", "blue", "orange", "green", "green", "gray"]
        assert colors_for_sequence(sequence) == expected

    def test_long_sequence(self):
        sequence = "ACGTU&" * 1000
        expected = ["red", "blue", "orange", "green", "green", "gray"] * 1000
        assert colors_for_sequence(sequence) == expected

    def test_with_spaces(self):
        with pytest.raises(ValueError) as exc_info:
            colors_for_sequence("AT CG")
        assert "Invalid character ' ' in sequence" in str(exc_info.value)
