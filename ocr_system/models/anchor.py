"""Configurações de âncoras para detecção de pontos de referência."""

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class QRCodeAnchorConfig(BaseModel):
    """Configuração para detecção de âncora via QR Code."""

    type: Literal["qrcode"] = "qrcode"
    expected_content_pattern: str | None = Field(
        default=None, description="Regex para validar conteúdo do QR code"
    )
    search_region: tuple[int, int, int, int] | None = Field(
        default=None, description="Região de busca opcional (x, y, width, height)"
    )
    expected_qr_size: tuple[int, int] | None = Field(
        default=None, description="Tamanho esperado do QR code (width, height) para cálculo de escala dos ROIs"
    )
    expected_qr_width_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Proporção esperada do QR code em relação à largura da imagem (0.0-1.0)"
    )
    expected_qr_height_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Proporção esperada do QR code em relação à altura da imagem (0.0-1.0)"
    )
    min_confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("search_region")
    @classmethod
    def validate_search_region(cls, v: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if v is not None:
            x, y, w, h = v
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError("search_region deve ter valores positivos e width/height > 0")
        return v

    @field_validator("expected_qr_size")
    @classmethod
    def validate_expected_qr_size(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None:
            w, h = v
            if w <= 0 or h <= 0:
                raise ValueError("expected_qr_size deve ter width e height > 0")
        return v


class TextAnchorConfig(BaseModel):
    """Configuração para detecção de âncora via texto."""

    type: Literal["text"] = "text"
    keyword: str = Field(description="Palavra-chave a buscar (ex: 'RESULTADO', 'LAUDO Nº')")
    search_region: tuple[int, int, int, int] | None = Field(
        default=None, description="Região de busca opcional (x, y, width, height)"
    )
    fuzzy_match: bool = Field(default=True, description="Usar fuzzy matching para tolerar erros")
    max_distance: int = Field(default=2, ge=0, description="Distância máxima de Levenshtein")

    @field_validator("search_region")
    @classmethod
    def validate_search_region(cls, v: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
        if v is not None:
            x, y, w, h = v
            if x < 0 or y < 0 or w <= 0 or h <= 0:
                raise ValueError("search_region deve ter valores positivos e width/height > 0")
        return v


class AnchorStrategy(BaseModel):
    """Estratégia de detecção de âncora com fallbacks."""

    primary: QRCodeAnchorConfig | TextAnchorConfig
    fallbacks: list[QRCodeAnchorConfig | TextAnchorConfig] = Field(default_factory=list)
    fallback_on_low_confidence: bool = Field(default=True)
    min_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    try_original_on_failure: bool = Field(
        default=True,
        description="Se True, tenta detectar na imagem original (sem pré-processamento) se falhar na processada"
    )


