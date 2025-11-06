"""Configurações de ROI (Region of Interest) para extração de campos."""

from typing import Literal
from pydantic import BaseModel, Field


class OCRConfig(BaseModel):
    """Configuração de OCR para um ROI específico."""

    type: Literal["text", "numeric", "date"] = Field(default="text")
    lang: str = Field(default="pt", description="Idioma para OCR")
    allowed_chars: str | None = Field(
        default=None, description="Caracteres permitidos (regex pattern)"
    )
    preprocessing: list[Literal["uppercase", "lowercase", "strip", "remove_spaces"]] = Field(
        default_factory=list
    )
    decimal_separator: str | None = Field(
        default=None, description="Separador decimal para campos numéricos"
    )
    expected_unit: str | None = Field(default=None, description="Unidade esperada (ex: 'g/dL')")
    formats: list[str] | None = Field(
        default=None, description="Formatos esperados (para tipo 'date')"
    )


class ValidationConfig(BaseModel):
    """Configuração de validação para um campo extraído."""

    regex: str | None = Field(default=None, description="Padrão regex para validar")
    range: tuple[float, float] | None = Field(
        default=None, description="Range válido para valores numéricos (min, max)"
    )
    required: bool = Field(default=True, description="Campo obrigatório")


class ROIConfig(BaseModel):
    """Configuração completa de um ROI."""

    id: str = Field(description="Identificador único do ROI")
    label: str = Field(description="Label descritivo do campo")
    relative_position: tuple[int, int] = Field(
        description="Posição relativa à âncora (offset_x, offset_y)"
    )
    size: tuple[int, int] = Field(description="Tamanho do ROI (width, height)")
    ocr_config: OCRConfig = Field(description="Configuração de OCR")
    validation: ValidationConfig = Field(description="Configuração de validação")

    @property
    def width(self) -> int:
        """Largura do ROI."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Altura do ROI."""
        return self.size[1]

