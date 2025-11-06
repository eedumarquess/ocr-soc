"""Testes do OCR Engine."""

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from ocr_system.core.ocr_engine import (
    extract_from_ocr_result,
    extract_text_from_roi,
    prepare_image_for_ocr,
)
from ocr_system.models.roi import OCRConfig
from ocr_system.utils.exceptions import OCRExtractionError

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def test_prepare_image_for_ocr_rgb() -> None:
    """Testa preparação de imagem RGB."""
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    result = prepare_image_for_ocr(image)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_prepare_image_for_ocr_grayscale() -> None:
    """Testa preparação de imagem grayscale."""
    image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    result = prepare_image_for_ocr(image)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_prepare_image_for_ocr_rgba() -> None:
    """Testa preparação de imagem RGBA."""
    image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
    result = prepare_image_for_ocr(image)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8


def test_prepare_image_for_ocr_float_normalization() -> None:
    """Testa normalização de imagem float para uint8."""
    image = np.random.rand(100, 100, 3).astype(np.float64) * 255
    result = prepare_image_for_ocr(image)
    assert result.shape == (100, 100, 3)
    assert result.dtype == np.uint8
    assert np.all(result >= 0)
    assert np.all(result <= 255)


def test_prepare_image_for_ocr_too_small() -> None:
    """Testa erro com imagem muito pequena."""
    image = np.ones((5, 5, 3), dtype=np.uint8)
    with pytest.raises(OCRExtractionError, match="muito pequena"):
        prepare_image_for_ocr(image)


def test_prepare_image_for_ocr_invalid_shape() -> None:
    """Testa erro com formato de imagem inválido."""
    image = np.ones((100,), dtype=np.uint8)
    with pytest.raises(OCRExtractionError, match="muito pequena ou formato inválido"):
        prepare_image_for_ocr(image)


def test_extract_from_ocr_result_dict_format() -> None:
    """Testa extração do formato novo (dict-like OCRResult)."""
    # Simula OCRResult com formato dict-like
    ocr_result = {
        'rec_texts': ['texto1', 'texto2'],
        'rec_scores': [0.9, 0.8],
        'dt_polys': [
            [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
            [[50.0, 60.0], [70.0, 60.0], [70.0, 80.0], [50.0, 80.0]],
        ],
    }
    
    texts, confidences, boxes = extract_from_ocr_result(ocr_result)
    
    assert texts == ['texto1', 'texto2']
    assert confidences == [0.9, 0.8]
    assert len(boxes) == 2
    assert boxes[0] == [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]]


def test_extract_from_ocr_result_list_format() -> None:
    """Testa extração do formato antigo (lista de listas)."""
    # Simula formato antigo: [[box, (text, confidence)], ...]
    ocr_result = [
        [
            [[10, 20], [30, 20], [30, 40], [10, 40]],
            ('texto1', 0.9),
        ],
        [
            [[50, 60], [70, 60], [70, 80], [50, 80]],
            ('texto2', 0.8),
        ],
    ]
    
    texts, confidences, boxes = extract_from_ocr_result(ocr_result)
    
    assert texts == ['texto1', 'texto2']
    assert confidences == [0.9, 0.8]
    assert len(boxes) == 2


def test_extract_from_ocr_result_numpy_arrays() -> None:
    """Testa extração com arrays numpy."""
    ocr_result = {
        'rec_texts': np.array(['texto1', 'texto2']),
        'rec_scores': np.array([0.9, 0.8]),
        'dt_polys': np.array([
            [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0], [10.0, 40.0]],
        ]),
    }
    
    texts, confidences, boxes = extract_from_ocr_result(ocr_result)
    
    assert texts == ['texto1', 'texto2']
    assert confidences == [0.9, 0.8]
    assert len(boxes) == 1


def test_extract_from_ocr_result_empty() -> None:
    """Testa extração com resultado vazio."""
    ocr_result = {}
    texts, confidences, boxes = extract_from_ocr_result(ocr_result)
    
    assert texts == []
    assert confidences == []
    assert boxes == []


def test_extract_from_ocr_result_none_values() -> None:
    """Testa extração com valores None."""
    ocr_result = {
        'rec_texts': ['texto1', None, 'texto2'],
        'rec_scores': [0.9, None, 0.8],
    }
    
    texts, confidences, boxes = extract_from_ocr_result(ocr_result)
    
    assert texts == ['texto1', 'texto2']
    assert confidences == [0.9, 0.8]


def test_extract_text_from_roi_success(mocker: "MockerFixture") -> None:
    """Testa extração de texto bem-sucedida."""
    # Mock do PaddleOCR
    mock_ocr = Mock()
    mock_ocr.ocr.return_value = [
        {
            'rec_texts': ['texto', 'extraído'],
            'rec_scores': [0.9, 0.8],
            'dt_polys': [],
        }
    ]
    
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    config = OCRConfig()
    
    text, confidence, metadata = extract_text_from_roi(image, config, mock_ocr)
    
    assert text == "texto extraído"
    assert abs(confidence - 0.85) < 0.001  # (0.9 + 0.8) / 2
    assert 'raw_text' in metadata
    assert 'confidence_scores' in metadata
    assert 'num_detections' in metadata
    assert metadata['num_detections'] == 2


def test_extract_text_from_roi_empty_result(mocker: "MockerFixture") -> None:
    """Testa extração com resultado vazio."""
    mock_ocr = Mock()
    mock_ocr.ocr.return_value = [None]
    
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    config = OCRConfig()
    
    text, confidence, metadata = extract_text_from_roi(image, config, mock_ocr)
    
    assert text == ""
    assert confidence == 0.0
    assert 'error' in metadata
    assert 'Nenhum texto detectado' in metadata['error']


def test_extract_text_from_roi_ocr_error(mocker: "MockerFixture") -> None:
    """Testa tratamento de erro do OCR."""
    mock_ocr = Mock()
    mock_ocr.ocr.side_effect = Exception("Erro no OCR")
    
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    config = OCRConfig()
    
    text, confidence, metadata = extract_text_from_roi(image, config, mock_ocr)
    
    assert text == ""
    assert confidence == 0.0
    assert 'error' in metadata
    assert 'Erro ao executar OCR' in metadata['error']


def test_extract_text_from_roi_image_preparation_error() -> None:
    """Testa tratamento de erro na preparação da imagem."""
    image = np.ones((5, 5, 3), dtype=np.uint8)  # Muito pequena
    config = OCRConfig()
    
    text, confidence, metadata = extract_text_from_roi(image, config, None)
    
    assert text == ""
    assert confidence == 0.0
    assert 'error' in metadata


def test_extract_text_from_roi_postprocessing(mocker: "MockerFixture") -> None:
    """Testa que pós-processamento é aplicado."""
    mock_ocr = Mock()
    mock_ocr.ocr.return_value = [
        {
            'rec_texts': ['  texto  ', 'com', 'espaços  '],
            'rec_scores': [0.9],
            'dt_polys': [],
        }
    ]
    
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    config = OCRConfig(preprocessing=["strip"])
    
    text, confidence, metadata = extract_text_from_roi(image, config, mock_ocr)
    
    # O pós-processamento deve remover espaços extras
    assert 'texto' in text
    assert 'com' in text
    assert 'espaços' in text

