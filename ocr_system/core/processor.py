"""Processador principal que orquestra todo o pipeline."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from ocr_system.core.anchor_detector import detect_anchor_with_fallback
from ocr_system.core.ocr_engine import initialize_ocr
from ocr_system.core.preprocessing import preprocess_pipeline
from ocr_system.core.roi_extractor import extract_all_rois
from ocr_system.models.layout import LayoutConfig
from ocr_system.models.result import FieldResult, OCRResult
from ocr_system.utils.exceptions import AnchorNotFoundError
from ocr_system.utils.image_io import load_image
from ocr_system.utils.medical_rules import normalize_medical_text
from ocr_system.utils.visualization import save_roi_visualization


def load_layout(layout_id: str, layouts_dir: Path | None = None) -> LayoutConfig:
    """
    Carrega layout JSON.

    Args:
        layout_id: ID do layout (nome do arquivo sem extensão)
        layouts_dir: Diretório de layouts (padrão: configs/layouts)

    Returns:
        Configuração do layout

    Raises:
        FileNotFoundError: Se layout não for encontrado
    """
    if layouts_dir is None:
        layouts_dir = Path(__file__).parent.parent.parent / "configs" / "layouts"

    layout_path = layouts_dir / f"{layout_id}.json"

    if not layout_path.exists():
        raise FileNotFoundError(f"Layout não encontrado: {layout_path}")

    with open(layout_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return LayoutConfig.model_validate(data)


def format_field_output(field_result: FieldResult, layout: LayoutConfig) -> dict[str, Any]:
    """
    Formata saída de um campo em formato JSON amigável.

    Args:
        field_result: Resultado da extração do campo
        layout: Configuração do layout (para obter a label)

    Returns:
        Dicionário com label, value, confiança e texto retornado
    """
    # Obtém a label do layout
    roi_config = layout.get_roi_by_id(field_result.roi_id)
    label = roi_config.label if roi_config else field_result.roi_id

    return {
        "label": label,
        "value": field_result.value if field_result.value else None,
        "confianca": round(field_result.confidence, 2) if field_result.confidence is not None else None,
        "texto_retornado": field_result.raw_text if field_result.raw_text else None,
        "valido": field_result.validated,
        "obrigatorio": field_result.metadata.get("required", False),
    }


def format_output_json(result: OCRResult, layout: LayoutConfig) -> dict[str, Any]:
    """
    Formata resultado completo em JSON amigável.

    Args:
        result: Resultado do processamento
        layout: Configuração do layout

    Returns:
        Dicionário JSON com todos os campos formatados
    """
    fields_output = [
        format_field_output(field, layout) for field in result.fields
    ]

    return {
        "layout_id": result.layout_id,
        "image_path": result.image_path,
        "anchor_detected": result.anchor_detected,
        "anchor_confidence": round(result.anchor_confidence, 2) if result.anchor_confidence is not None else None,
        "campos": fields_output,
        "erros": result.errors,
    }


def process_document(
    image_path: str | Path,
    layout: LayoutConfig,
    debug_dir: Path | None = None,
) -> OCRResult:
    """
    Processa um documento completo.

    Args:
        image_path: Caminho da imagem
        layout: Configuração do layout
        debug_dir: Diretório para salvar imagens intermediárias (opcional)

    Returns:
        Resultado completo do processamento
    """
    image_path = Path(image_path)
    errors: list[str] = []
    processing_metadata: dict[str, Any] = {}

    # Carrega imagem
    try:
        image = load_image(image_path)
    except Exception as e:
        return OCRResult(
            layout_id=layout.layout_id,
            image_path=str(image_path),
            anchor_detected=False,
            errors=[f"Erro ao carregar imagem: {str(e)}"],
        )

    # Pré-processamento
    try:
        processed_image, preprocess_metadata = preprocess_pipeline(image, layout.preprocessing, debug_dir)
        processing_metadata["preprocessing"] = preprocess_metadata
    except Exception as e:
        errors.append(f"Erro no pré-processamento: {str(e)}")
        processed_image = image  # Usa imagem original como fallback

    # Detecta âncora
    anchor_position: tuple[int, int] | None = None
    anchor_confidence: float | None = None
    anchor_type: str | None = None
    anchor_metadata: dict[str, Any] = {}

    ocr_instance = initialize_ocr()
    
    # Detecta âncora usando função compartilhada com fallback configurável
    try:
        position, confidence, a_type, metadata = detect_anchor_with_fallback(
            processed_image, image, layout.anchor, ocr_instance
        )
        anchor_position = position
        anchor_confidence = float(confidence) if confidence is not None else None
        anchor_type = a_type
        # Limpa metadados de tipos numpy
        anchor_metadata = {}
        if metadata:
            import numpy as np
            for key, value in metadata.items():
                if isinstance(value, (np.floating, np.integer)):
                    anchor_metadata[key] = float(value)
                elif isinstance(value, (list, tuple)):
                    anchor_metadata[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
                else:
                    anchor_metadata[key] = value
    except AnchorNotFoundError as e:
        errors.append(f"Âncora não detectada: {str(e)}")
        # Modo degradado: usa coordenadas absolutas (0, 0) como fallback
        anchor_position = (0, 0)
        anchor_confidence = 0.0
        anchor_type = "degraded"
        processing_metadata["degraded_mode"] = True

    # Extrai ROIs
    fields: list[Any] = []
    try:
        if anchor_position:
            fields = extract_all_rois(processed_image, anchor_position, layout, ocr_instance)
    except Exception as e:
        errors.append(f"Erro ao extrair ROIs: {str(e)}")

    # Aplica pós-processamento
    if layout.post_processing.medical_term_normalization:
        for field in fields:
            if field.value:
                field.value = normalize_medical_text(field.value)

    # Cria resultado
    result = OCRResult(
        layout_id=layout.layout_id,
        image_path=str(image_path),
        anchor_detected=anchor_position is not None and anchor_type != "degraded",
        anchor_type=anchor_type,
        anchor_position=anchor_position,
        anchor_confidence=anchor_confidence,
        fields=fields,
        processing_metadata={**processing_metadata, "anchor_metadata": anchor_metadata},
        errors=errors,
    )

    # Salva visualização se debug_dir estiver definido
    if debug_dir:
        visualization_path = debug_dir / "06_rois_visualized.jpg"
        save_roi_visualization(processed_image, result, layout, visualization_path, show_values=True)

    return result

