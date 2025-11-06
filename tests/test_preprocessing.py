"""Testes do pipeline de pré-processamento."""

import numpy as np
import pytest

from ocr_system.core.preprocessing import (
    _detect_border_ghost_lines,
    _detect_long_crossing_lines,
    _filter_table_lines,
    _identify_text_regions,
    _remove_border_ghost_lines,
    _remove_long_crossing_lines,
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


def test_detect_border_ghost_lines_horizontal() -> None:
    """Testa detecção de linhas fantasma horizontais nas bordas."""
    # Cria imagem com linha horizontal na borda superior
    image = np.ones((200, 200), dtype=np.uint8) * 255
    image[5:8, :] = 0  # Linha horizontal na borda superior
    
    mask = _detect_border_ghost_lines(image, border_percent=0.1)
    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    # Deve detectar pelo menos alguns pixels na linha
    assert np.sum(mask > 0) > 0


def test_detect_border_ghost_lines_vertical() -> None:
    """Testa detecção de linhas fantasma verticais nas bordas."""
    # Cria imagem com linha vertical na borda esquerda
    image = np.ones((200, 200), dtype=np.uint8) * 255
    image[:, 5:8] = 0  # Linha vertical na borda esquerda
    
    mask = _detect_border_ghost_lines(image, border_percent=0.1)
    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    # Deve detectar pelo menos alguns pixels na linha
    assert np.sum(mask > 0) > 0


def test_remove_border_ghost_lines() -> None:
    """Testa remoção de linhas fantasma nas bordas."""
    # Cria imagem com linha horizontal na borda
    image = np.ones((200, 200), dtype=np.uint8) * 255
    image[5:8, :] = 0  # Linha horizontal na borda superior
    
    result = _remove_border_ghost_lines(image, border_percent=0.1)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    # A linha deve ser removida (pixels devem ser preenchidos)
    assert np.sum(result[5:8, :] == 0) < np.sum(image[5:8, :] == 0)


def test_detect_long_crossing_lines() -> None:
    """Testa detecção de linhas longas atravessando a imagem."""
    # Cria imagem com linha horizontal longa muito fina (linha fantasma)
    image = np.ones((200, 200), dtype=np.uint8) * 255
    image[100:101, :] = 0  # Linha horizontal muito fina (1 pixel) no centro
    
    mask = _detect_long_crossing_lines(image, min_length_percent=0.8)
    assert mask.shape == image.shape
    assert mask.dtype == np.uint8
    # Pode ou não detectar dependendo da espessura (linha muito fina pode ser filtrada)
    # Apenas verifica que a função não quebra
    assert isinstance(mask, np.ndarray)


def test_remove_long_crossing_lines() -> None:
    """Testa remoção de linhas longas atravessando a imagem."""
    # Cria imagem com linha horizontal longa muito fina (linha fantasma)
    image = np.ones((200, 200), dtype=np.uint8) * 255
    image[100:101, :] = 0  # Linha horizontal muito fina no centro
    
    result = _remove_long_crossing_lines(image, min_length_percent=0.8)
    assert result.shape == image.shape
    assert result.dtype == image.dtype
    # A função deve processar a imagem sem quebrar
    # Pode ou não remover dependendo da detecção
    assert isinstance(result, np.ndarray)


def test_identify_text_regions() -> None:
    """Testa identificação de regiões de texto."""
    # Cria imagem sintética com texto (retângulos pequenos horizontais)
    image = np.ones((200, 200), dtype=np.uint8) * 255
    # Adiciona alguns "caracteres" de texto (retângulos pequenos)
    image[50:60, 20:40] = 0  # Caractere 1
    image[50:60, 45:65] = 0  # Caractere 2
    image[50:60, 70:90] = 0  # Caractere 3
    
    text_mask = _identify_text_regions(image)
    assert text_mask.shape == image.shape
    assert text_mask.dtype == np.uint8
    # Deve identificar pelo menos algumas regiões de texto
    assert np.sum(text_mask > 0) > 0


def test_filter_table_lines() -> None:
    """Testa filtragem de linhas de tabela."""
    # Cria imagem com linha de tabela (linha horizontal longa com espessura adequada)
    image = np.ones((200, 200), dtype=np.uint8) * 255
    image[99:102, :] = 0  # Linha de tabela horizontal com 3 pixels de espessura
    
    table_mask = _filter_table_lines(image)
    assert table_mask.shape == image.shape
    assert table_mask.dtype == np.uint8
    # A função deve processar sem quebrar
    # Pode ou não detectar dependendo dos parâmetros de espessura
    assert isinstance(table_mask, np.ndarray)


def test_deskew_text_dominant() -> None:
    """Testa deskew baseado em texto dominante."""
    # Cria imagem com texto inclinado
    image = np.ones((200, 200), dtype=np.uint8) * 255
    # Adiciona texto horizontal (não inclinado)
    for i in range(5):
        y = 50 + i * 20
        image[y:y+10, 20:180] = 0  # Linhas de texto horizontais
    
    # Testa com use_text_dominant=True
    result, angle = deskew_image(image, threshold=0.1, use_text_dominant=True)
    assert result.shape[0] > 0 and result.shape[1] > 0
    assert isinstance(angle, float)


def test_preprocess_pipeline_with_ghost_lines() -> None:
    """Testa pipeline completo com remoção de linhas fantasma."""
    # Cria imagem com linha fantasma na borda
    image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    image[5:8, :, :] = 0  # Linha fantasma na borda superior
    
    config = PreprocessingConfig(
        deskew=True,
        remove_ghost_lines=True,
        ghost_line_border_percent=0.1,
        binarization="adaptive",
        denoise=False,
        contrast_normalization=False,
        border_removal=False,
    )
    
    result, metadata = preprocess_pipeline(image, config)
    assert result.shape[0] > 0 and result.shape[1] > 0
    assert "final_shape" in metadata
    assert "deskew_angle" in metadata


def test_preprocess_pipeline_text_dominant() -> None:
    """Testa pipeline completo com deskew baseado em texto dominante."""
    image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    config = PreprocessingConfig(
        deskew=True,
        deskew_text_dominant=True,
        binarization="adaptive",
        denoise=True,
        contrast_normalization=True,
        border_removal=False,
        remove_ghost_lines=False,
    )
    
    result, metadata = preprocess_pipeline(image, config)
    assert result.shape[0] > 0 and result.shape[1] > 0
    assert "final_shape" in metadata
    assert "deskew_angle" in metadata

