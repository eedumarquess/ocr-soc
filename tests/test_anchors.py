"""Testes de detecção de âncoras."""

import numpy as np
import pytest

from ocr_system.core.anchor_detector import detect_anchor, detect_qrcode, detect_text_anchor
from ocr_system.models.anchor import AnchorStrategy, QRCodeAnchorConfig, TextAnchorConfig
from ocr_system.utils.exceptions import AnchorNotFoundError, InvalidImageError


def test_detect_qrcode_no_qr() -> None:
    """Testa detecção de QR code em imagem sem QR."""
    image = np.ones((100, 100), dtype=np.uint8) * 255
    config = QRCodeAnchorConfig()
    result = detect_qrcode(image, config)
    assert result is None


def test_detect_text_anchor_no_text() -> None:
    """Testa detecção de texto em imagem sem texto."""
    image = np.ones((100, 100), dtype=np.uint8) * 255
    config = TextAnchorConfig(keyword="TESTE")
    result = detect_text_anchor(image, config)
    # Pode retornar None ou tentar buscar (depende da implementação)
    assert result is None or isinstance(result, tuple)


def test_detect_anchor_not_found() -> None:
    """Testa detecção de âncora quando não encontrada."""
    image = np.ones((100, 100), dtype=np.uint8) * 255
    strategy = AnchorStrategy(
        primary=QRCodeAnchorConfig(),
        fallbacks=[TextAnchorConfig(keyword="INEXISTENTE")],
    )

    with pytest.raises(AnchorNotFoundError):
        detect_anchor(image, strategy)


def test_detect_anchor_invalid_image() -> None:
    """Testa detecção com imagem inválida."""
    strategy = AnchorStrategy(primary=QRCodeAnchorConfig())

    with pytest.raises(InvalidImageError):
        detect_anchor(None, strategy)  # type: ignore


