"""Modelos de resultado do OCR."""

import numpy as np
from typing import Any
from pydantic import BaseModel, Field, field_serializer


class FieldResult(BaseModel):
    """Resultado da extração de um campo individual."""

    roi_id: str = Field(description="ID do ROI extraído")
    value: str | None = Field(default=None, description="Valor extraído")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Confiança do OCR")
    raw_text: str | None = Field(default=None, description="Texto bruto antes do pós-processamento")
    validated: bool = Field(default=False, description="Se passou na validação")
    validation_error: str | None = Field(default=None, description="Erro de validação se houver")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais")

    @field_serializer("metadata")
    def serialize_metadata(self, value: dict[str, Any]) -> dict[str, Any]:
        """Serializa metadados convertendo tipos numpy."""
        if not value:
            return value
        result = {}
        for key, val in value.items():
            if isinstance(val, (np.floating, np.integer)):
                result[key] = float(val)
            elif isinstance(val, (list, tuple)):
                result[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in val
                ]
            elif isinstance(val, dict):
                result[key] = self.serialize_metadata(val)
            else:
                result[key] = val
        return result


class OCRResult(BaseModel):
    """Resultado completo do processamento de um documento."""

    layout_id: str = Field(description="ID do layout utilizado")
    image_path: str = Field(description="Caminho da imagem processada")
    anchor_detected: bool = Field(description="Se a âncora foi detectada com sucesso")
    anchor_type: str | None = Field(default=None, description="Tipo de âncora detectada")
    anchor_position: tuple[int, int] | None = Field(
        default=None, description="Posição da âncora (x, y)"
    )
    anchor_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confiança da detecção da âncora"
    )
    fields: list[FieldResult] = Field(default_factory=list, description="Campos extraídos")
    processing_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadados do processamento"
    )
    errors: list[str] = Field(default_factory=list, description="Erros encontrados durante processamento")

    @field_serializer("processing_metadata")
    def serialize_processing_metadata(self, value: dict[str, Any]) -> dict[str, Any]:
        """Serializa metadados de processamento convertendo tipos numpy."""
        if not value:
            return value
        result = {}
        for key, val in value.items():
            if isinstance(val, (np.floating, np.integer)):
                result[key] = float(val)
            elif isinstance(val, (list, tuple)):
                result[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for v in val
                ]
            elif isinstance(val, dict):
                result[key] = self.serialize_processing_metadata(val)
            else:
                result[key] = val
        return result

    @property
    def success_rate(self) -> float:
        """Taxa de sucesso (campos validados / campos obrigatórios)."""
        if not self.fields:
            return 0.0
        required_fields = [f for f in self.fields if f.metadata.get("required", True)]
        if not required_fields:
            return 1.0
        validated = sum(1 for f in required_fields if f.validated)
        return validated / len(required_fields)

