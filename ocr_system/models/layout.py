"""Schema completo do layout JSON."""

from typing import Any, Literal
from pydantic import BaseModel, Field

from ocr_system.models.anchor import AnchorStrategy
from ocr_system.models.roi import ROIConfig


class PreprocessingConfig(BaseModel):
    """Configuração do pipeline de pré-processamento."""

    deskew: bool = Field(default=True, description="Aplicar correção de rotação")
    deskew_threshold: float = Field(default=0.5, description="Graus mínimos para corrigir")
    binarization: Literal["otsu", "adaptive", "none"] = Field(
        default="adaptive", description="Tipo de binarização"
    )
    adaptive_block_size: int = Field(default=15, ge=3, description="Tamanho do bloco para binarização adaptativa")
    denoise: bool = Field(default=True, description="Aplicar redução de ruído")
    denoise_strength: int = Field(default=3, ge=1, le=20, description="Força da redução de ruído")
    sharpen: bool = Field(default=True, description="Aplicar sharpening após denoise para recuperar nitidez")
    sharpen_strength: float = Field(default=0.5, ge=0.1, le=2.0, description="Força do sharpening (unsharp mask)")
    contrast_normalization: bool = Field(default=True, description="Normalizar contraste (CLAHE)")
    clahe_clip_limit: float = Field(default=1.2, ge=0.1, le=10.0, description="Limite de clip para CLAHE")
    border_removal: bool = Field(default=True, description="Remover bordas pretas")
    border_threshold: int = Field(default=10, ge=1, description="Threshold de pixels de borda preta")
    resize_to_reference: bool = Field(default=False, description="Redimensionar para DPI de referência")
    reference_dpi: int = Field(default=300, ge=72, description="DPI de referência")


class PostProcessingConfig(BaseModel):
    """Configuração de pós-processamento."""

    medical_term_normalization: bool = Field(
        default=True, description="Normalizar termos médicos"
    )
    decimal_standardization: str = Field(
        default=".", description="Separador decimal padrão (., ou ,)"
    )
    output_format: Literal["json", "csv"] = Field(default="json", description="Formato de saída")


class LayoutConfig(BaseModel):
    """Configuração completa de um layout."""

    layout_id: str = Field(description="Identificador único do layout")
    description: str = Field(description="Descrição do tipo de documento")
    version: str = Field(description="Versão do layout")
    created_at: str = Field(description="Data de criação (ISO format)")
    preprocessing: PreprocessingConfig = Field(
        default_factory=PreprocessingConfig, description="Configuração de pré-processamento"
    )
    anchor: AnchorStrategy = Field(description="Estratégia de detecção de âncora")
    rois: list[ROIConfig] = Field(description="Lista de ROIs a extrair")
    post_processing: PostProcessingConfig = Field(
        default_factory=PostProcessingConfig, description="Configuração de pós-processamento"
    )

    def get_roi_by_id(self, roi_id: str) -> ROIConfig | None:
        """Retorna ROI por ID."""
        for roi in self.rois:
            if roi.id == roi_id:
                return roi
        return None

    def get_required_rois(self) -> list[ROIConfig]:
        """Retorna apenas ROIs obrigatórios."""
        return [roi for roi in self.rois if roi.validation.required]

