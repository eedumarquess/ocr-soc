"""Testes do pipeline de pré-processamento."""

import numpy as np
import pytest

from ocr_system.core.preprocessing import (
    binarize_image,
    denoise_image,
    deskew_image,
    normalize_contrast,
    preprocess_pipeline,
    remove_borders,
)
from ocr_system.models.layout import PreprocessingConfig
from ocr_system.utils.exceptions import InvalidImageError


def test_deskew_image_no_rotation() -> None:
    """Testa deskew em imagem sem rotação."""
    image = np.ones((100, 100), dtype=np.uint8) * 255
    result, angle = deskew_image(image, threshold=0.5)
    assert result.shape == image.shape
    assert abs(angle) < 1.0  # Deve detectar ângulo próximo de zero


def test_binarize_image_otsu() -> None:
    """Testa binarização Otsu."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = binarize_image(image, method="otsu")
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.all((result == 0) | (result == 255))


def test_binarize_image_adaptive() -> None:
    """Testa binarização adaptativa."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = binarize_image(image, method="adaptive", block_size=11)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_binarize_image_none() -> None:
    """Testa binarização desabilitada."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = binarize_image(image, method="none")
    assert np.array_equal(result, image)


def test_denoise_image() -> None:
    """Testa redução de ruído."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = denoise_image(image, strength=10)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_normalize_contrast() -> None:
    """Testa normalização de contraste."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = normalize_contrast(image, clip_limit=2.0)
    assert result.shape == image.shape
    assert result.dtype == image.dtype


def test_remove_borders() -> None:
    """Testa remoção de bordas."""
    # Cria imagem com bordas pretas
    image = np.ones((100, 100), dtype=np.uint8) * 255
    image[0:10, :] = 0  # Borda superior
    image[:, 0:10] = 0  # Borda esquerda
    image[90:, :] = 0  # Borda inferior
    image[:, 90:] = 0  # Borda direita

    result = remove_borders(image, threshold=10)
    assert result.shape[0] < image.shape[0] or result.shape[1] < image.shape[1]


def test_preprocess_pipeline() -> None:
    """Testa pipeline completo."""
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    config = PreprocessingConfig(
        deskew=True,
        binarization="adaptive",
        denoise=True,
        contrast_normalization=True,
        border_removal=False,
    )

    result, metadata = preprocess_pipeline(image, config)
    assert result.shape[0] > 0 and result.shape[1] > 0
    assert "final_shape" in metadata


def test_preprocess_pipeline_invalid_image() -> None:
    """Testa pipeline com imagem inválida."""
    with pytest.raises(InvalidImageError):
        preprocess_pipeline(None, PreprocessingConfig())  # type: ignore


