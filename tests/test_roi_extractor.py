"""Testes de extração de ROIs."""

import numpy as np
import pytest

from ocr_system.core.roi_extractor import extract_roi_field, validate_field
from ocr_system.models.roi import OCRConfig, ROIConfig, ValidationConfig
from ocr_system.utils.exceptions import InvalidImageError


def test_validate_field_required_empty() -> None:
    """Testa validação de campo obrigatório vazio."""
    roi_config = ROIConfig(
        id="test",
        label="Test",
        relative_position=(0, 0),
        size=(100, 50),
        ocr_config=OCRConfig(),
        validation=ValidationConfig(required=True),
    )

    is_valid, error = validate_field("", roi_config)
    assert not is_valid
    assert error is not None


def test_validate_field_optional_empty() -> None:
    """Testa validação de campo opcional vazio."""
    roi_config = ROIConfig(
        id="test",
        label="Test",
        relative_position=(0, 0),
        size=(100, 50),
        ocr_config=OCRConfig(),
        validation=ValidationConfig(required=False),
    )

    is_valid, error = validate_field("", roi_config)
    assert is_valid
    assert error is None


def test_validate_field_regex() -> None:
    """Testa validação com regex."""
    roi_config = ROIConfig(
        id="test",
        label="Test",
        relative_position=(0, 0),
        size=(100, 50),
        ocr_config=OCRConfig(),
        validation=ValidationConfig(regex=r"^\d+$", required=True),
    )

    is_valid, _ = validate_field("123", roi_config)
    assert is_valid

    is_valid, _ = validate_field("abc", roi_config)
    assert not is_valid


def test_validate_field_range() -> None:
    """Testa validação de range numérico."""
    roi_config = ROIConfig(
        id="test",
        label="Test",
        relative_position=(0, 0),
        size=(100, 50),
        ocr_config=OCRConfig(type="numeric"),
        validation=ValidationConfig(range=(0.0, 100.0), required=True),
    )

    is_valid, _ = validate_field("50", roi_config)
    assert is_valid

    is_valid, _ = validate_field("150", roi_config)
    assert not is_valid


def test_extract_roi_field_invalid_image() -> None:
    """Testa extração com imagem inválida."""
    roi_config = ROIConfig(
        id="test",
        label="Test",
        relative_position=(0, 0),
        size=(100, 50),
        ocr_config=OCRConfig(),
        validation=ValidationConfig(),
    )

    with pytest.raises(InvalidImageError):
        extract_roi_field(None, (0, 0), roi_config)  # type: ignore


