"""Testes end-to-end do sistema."""

import json
from pathlib import Path

import pytest

from ocr_system.core.processor import load_layout, process_document


def test_load_layout_not_found() -> None:
    """Testa carregamento de layout inexistente."""
    with pytest.raises(FileNotFoundError):
        load_layout("inexistente")


def test_process_document_invalid_path() -> None:
    """Testa processamento com caminho inválido."""
    # Cria layout temporário para teste
    layout_data = {
        "layout_id": "test",
        "description": "Test layout",
        "version": "1.0",
        "created_at": "2025-01-01T00:00:00Z",
        "preprocessing": {},
        "anchor": {
            "primary": {"type": "qrcode"},
            "fallbacks": [],
        },
        "rois": [],
        "post_processing": {},
    }

    # Salva layout temporário
    layouts_dir = Path(__file__).parent.parent / "configs" / "layouts"
    layouts_dir.mkdir(parents=True, exist_ok=True)
    test_layout_path = layouts_dir / "test.json"

    with open(test_layout_path, "w", encoding="utf-8") as f:
        json.dump(layout_data, f)

    try:
        layout = load_layout("test")
        result = process_document("arquivo_inexistente.jpg", layout)

        # Deve retornar resultado com erro
        assert not result.anchor_detected
        assert len(result.errors) > 0

    finally:
        # Remove layout de teste
        if test_layout_path.exists():
            test_layout_path.unlink()


