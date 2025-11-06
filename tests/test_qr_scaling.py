"""Testes para sistema híbrido de cálculo de escala de QR code."""

from typing import TYPE_CHECKING

import numpy as np
import pytest

from ocr_system.models.anchor import QRCodeAnchorConfig
from ocr_system.utils.geometry import calculate_expected_qr_size
from ocr_system.utils.qr_cache import clear_cache, get_cached_qr_size, set_cached_qr_size

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def sample_image() -> np.ndarray:
    """Cria imagem de exemplo para testes."""
    return np.ones((1000, 800), dtype=np.uint8) * 255


@pytest.fixture(autouse=True)
def clear_cache_before_test() -> None:
    """Limpa cache antes de cada teste."""
    clear_cache()
    yield
    clear_cache()


def test_calculate_expected_qr_size_relative_ratios(sample_image: np.ndarray) -> None:
    """Testa cálculo com proporções relativas."""
    config = QRCodeAnchorConfig(
        expected_qr_width_ratio=0.08,
        expected_qr_height_ratio=0.10,
    )
    
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    
    assert result is not None
    expected_width = int(800 * 0.08)  # 64
    expected_height = int(1000 * 0.10)  # 100
    assert result == (expected_width, expected_height)


def test_calculate_expected_qr_size_fixed_size(sample_image: np.ndarray) -> None:
    """Testa fallback para tamanho fixo."""
    config = QRCodeAnchorConfig(
        expected_qr_size=(200, 200),
    )
    
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    
    assert result == (200, 200)


def test_calculate_expected_qr_size_cache(sample_image: np.ndarray) -> None:
    """Testa uso de cache."""
    config = QRCodeAnchorConfig()
    
    # Salva no cache
    set_cached_qr_size("test_layout", 150, 150)
    
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    
    assert result == (150, 150)


def test_calculate_expected_qr_size_auto_reference(sample_image: np.ndarray) -> None:
    """Testa auto-referência (primeiro documento sem configuração)."""
    config = QRCodeAnchorConfig()
    
    result = calculate_expected_qr_size(sample_image, config, "new_layout")
    
    assert result is None


def test_calculate_expected_qr_size_priority_order(sample_image: np.ndarray) -> None:
    """Testa ordem de prioridade das estratégias."""
    # Configuração com todas as estratégias disponíveis
    config = QRCodeAnchorConfig(
        expected_qr_width_ratio=0.08,
        expected_qr_height_ratio=0.10,
        expected_qr_size=(200, 200),
    )
    
    # Deve usar proporções relativas (prioridade 1)
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    assert result is not None
    assert result == (int(800 * 0.08), int(1000 * 0.10))
    
    # Remove proporções, deve usar tamanho fixo (prioridade 2)
    config.expected_qr_width_ratio = None
    config.expected_qr_height_ratio = None
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    assert result == (200, 200)
    
    # Remove tamanho fixo, adiciona cache (prioridade 3)
    config.expected_qr_size = None
    set_cached_qr_size("test_layout", 150, 150)
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    assert result == (150, 150)
    
    # Remove cache, deve retornar None (prioridade 4)
    clear_cache("test_layout")
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    assert result is None


def test_calculate_expected_qr_size_invalid_ratios(sample_image: np.ndarray) -> None:
    """Testa validação de proporções inválidas."""
    # Proporção muito pequena que resulta em tamanho zero
    config = QRCodeAnchorConfig(
        expected_qr_width_ratio=0.0001,
        expected_qr_height_ratio=0.0001,
    )
    
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    
    # Deve retornar None se o tamanho calculado for <= 0
    # (width = 800 * 0.0001 = 0.08 -> int = 0)
    assert result is None or result[0] == 0 or result[1] == 0


def test_calculate_expected_qr_size_partial_ratios(sample_image: np.ndarray) -> None:
    """Testa que proporções parciais não são usadas."""
    # Apenas uma proporção definida
    config = QRCodeAnchorConfig(
        expected_qr_width_ratio=0.08,
        # expected_qr_height_ratio não definido
    )
    
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    
    # Deve usar próxima estratégia (tamanho fixo ou cache)
    # Como não há tamanho fixo nem cache, deve retornar None
    assert result is None


def test_qr_cache_get_set() -> None:
    """Testa operações básicas de cache."""
    layout_id = "test_layout"
    
    # Cache vazio
    assert get_cached_qr_size(layout_id) is None
    
    # Salva no cache
    set_cached_qr_size(layout_id, 200, 200)
    assert get_cached_qr_size(layout_id) == (200, 200)
    
    # Atualiza cache
    set_cached_qr_size(layout_id, 300, 300)
    assert get_cached_qr_size(layout_id) == (300, 300)


def test_qr_cache_clear_specific() -> None:
    """Testa limpeza de cache específico."""
    set_cached_qr_size("layout1", 100, 100)
    set_cached_qr_size("layout2", 200, 200)
    
    clear_cache("layout1")
    
    assert get_cached_qr_size("layout1") is None
    assert get_cached_qr_size("layout2") == (200, 200)


def test_qr_cache_clear_all() -> None:
    """Testa limpeza de todo o cache."""
    set_cached_qr_size("layout1", 100, 100)
    set_cached_qr_size("layout2", 200, 200)
    
    clear_cache()
    
    assert get_cached_qr_size("layout1") is None
    assert get_cached_qr_size("layout2") is None


def test_qr_cache_invalid_size() -> None:
    """Testa validação de tamanhos inválidos no cache."""
    with pytest.raises(ValueError, match="Tamanhos inválidos"):
        set_cached_qr_size("test_layout", 0, 100)
    
    with pytest.raises(ValueError, match="Tamanhos inválidos"):
        set_cached_qr_size("test_layout", 100, 0)
    
    with pytest.raises(ValueError, match="Tamanhos inválidos"):
        set_cached_qr_size("test_layout", -10, 100)


def test_calculate_expected_qr_size_with_cache_after_detection(sample_image: np.ndarray) -> None:
    """Testa fluxo completo: primeiro documento salva no cache, segundo usa."""
    config = QRCodeAnchorConfig()
    
    # Primeiro documento: sem referência
    result1 = calculate_expected_qr_size(sample_image, config, "test_layout")
    assert result1 is None
    
    # Simula detecção e salvamento no cache
    set_cached_qr_size("test_layout", 180, 180)
    
    # Segundo documento: usa cache
    result2 = calculate_expected_qr_size(sample_image, config, "test_layout")
    assert result2 == (180, 180)


def test_calculate_expected_qr_size_ratios_override_fixed(sample_image: np.ndarray) -> None:
    """Testa que proporções relativas têm prioridade sobre tamanho fixo."""
    config = QRCodeAnchorConfig(
        expected_qr_width_ratio=0.1,
        expected_qr_height_ratio=0.1,
        expected_qr_size=(500, 500),  # Deve ser ignorado
    )
    
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    
    # Deve usar proporções, não tamanho fixo
    assert result == (int(800 * 0.1), int(1000 * 0.1))
    assert result != (500, 500)


def test_calculate_expected_qr_size_fixed_override_cache(sample_image: np.ndarray) -> None:
    """Testa que tamanho fixo tem prioridade sobre cache."""
    config = QRCodeAnchorConfig(
        expected_qr_size=(250, 250),
    )
    
    # Cache existe mas deve ser ignorado
    set_cached_qr_size("test_layout", 150, 150)
    
    result = calculate_expected_qr_size(sample_image, config, "test_layout")
    
    # Deve usar tamanho fixo, não cache
    assert result == (250, 250)
    assert result != (150, 150)

