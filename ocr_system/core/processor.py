"""Processador principal que orquestra todo o pipeline."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ocr_system.core.anchor_detector import detect_anchor_with_fallback
from ocr_system.core.ocr_engine import initialize_ocr
from ocr_system.core.preprocessing import preprocess_pipeline
from ocr_system.core.roi_extractor import extract_all_rois
from ocr_system.models.anchor import QRCodeAnchorConfig
from ocr_system.models.layout import LayoutConfig
from ocr_system.models.result import FieldResult, OCRResult
from ocr_system.utils.exceptions import AnchorNotFoundError
from ocr_system.utils.geometry import calculate_expected_qr_size
from ocr_system.utils.image_io import load_image
from ocr_system.utils.medical_rules import normalize_medical_text
from ocr_system.utils.qr_cache import set_cached_qr_size
from ocr_system.utils.visualization import save_roi_visualization

logger = logging.getLogger(__name__)


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

    # Calcula fator de escala baseado no tamanho do QR code detectado usando sistema híbrido
    scale_factor: float | tuple[float, float] = 1.0
    scaling_strategy = "none"
    
    if anchor_type == "qrcode" and anchor_metadata:
        # Verifica se temos tamanho do QR code detectado
        qr_width = anchor_metadata.get("qr_width")
        qr_height = anchor_metadata.get("qr_height")
        
        # Verifica se temos configuração de QR code na primary anchor
        if isinstance(layout.anchor.primary, QRCodeAnchorConfig):
            primary_config = layout.anchor.primary
            
            try:
                # Obtém tamanho esperado usando sistema híbrido
                expected_size = calculate_expected_qr_size(
                    processed_image, primary_config, layout.layout_id
                )
                
                if expected_size is not None:
                    expected_width, expected_height = expected_size
                    if qr_width and qr_height and expected_width > 0 and expected_height > 0:
                        # Calcula escalas separadas para X e Y (mais preciso que média)
                        # Usa divisão de ponto flutuante com alta precisão
                        scale_x = float(qr_width) / float(expected_width)
                        scale_y = float(qr_height) / float(expected_height)
                        # Arredonda para 6 casas decimais para manter precisão
                        scale_x = round(scale_x, 6)
                        scale_y = round(scale_y, 6)
                        scale_factor = (scale_x, scale_y)
                        
                        # Determina qual estratégia foi usada
                        if primary_config.expected_qr_width_ratio is not None and primary_config.expected_qr_height_ratio is not None:
                            scaling_strategy = "relative_ratios"
                        elif primary_config.expected_qr_size is not None:
                            scaling_strategy = "fixed_size"
                        else:
                            scaling_strategy = "cache"
                        
                        # Armazena ambos os fatores para compatibilidade
                        scale_avg = (scale_x + scale_y) / 2.0
                        processing_metadata["scale_factor"] = scale_avg
                        processing_metadata["scale_factor_x"] = scale_x
                        processing_metadata["scale_factor_y"] = scale_y
                        processing_metadata["qr_detected_size"] = (qr_width, qr_height)
                        processing_metadata["qr_expected_size"] = (expected_width, expected_height)
                        processing_metadata["scaling_strategy"] = scaling_strategy
                        
                        logger.info(
                            f"Escala calculada usando estratégia '{scaling_strategy}': "
                            f"detectado=({qr_width}, {qr_height}), "
                            f"esperado=({expected_width}, {expected_height}), "
                            f"fator_x={scale_x:.3f}, fator_y={scale_y:.3f}, média={scale_avg:.3f}"
                        )
                    else:
                        logger.warning(
                            f"Tamanhos inválidos para cálculo de escala: "
                            f"detectado=({qr_width}, {qr_height}), "
                            f"esperado=({expected_width}, {expected_height})"
                        )
                else:
                    # Auto-referência: usar QR detectado como referência (escala 1.0)
                    scaling_strategy = "auto_reference"
                    if qr_width and qr_height:
                        # Salva no cache para próximos documentos
                        try:
                            set_cached_qr_size(layout.layout_id, qr_width, qr_height)
                            logger.info(
                                f"QR code usado como referência (escala 1.0) e salvo no cache: "
                                f"({qr_width}, {qr_height})"
                            )
                        except ValueError as e:
                            logger.warning(f"Erro ao salvar no cache: {e}")
                    
                    processing_metadata["scaling_strategy"] = scaling_strategy
                    processing_metadata["qr_detected_size"] = (qr_width, qr_height)
                    logger.info(
                        f"Nenhuma referência disponível, usando QR detectado como referência "
                        f"(escala 1.0): ({qr_width}, {qr_height})"
                    )
            except Exception as e:
                logger.error(f"Erro ao calcular tamanho esperado do QR code: {e}", exc_info=True)
                # Fallback para escala 1.0
                scale_factor = 1.0
                scaling_strategy = "error_fallback"
                processing_metadata["scaling_strategy"] = scaling_strategy
                processing_metadata["scaling_error"] = str(e)
        else:
            # Se QR code foi detectado mas não há configuração, salva no cache
            if qr_width and qr_height:
                try:
                    set_cached_qr_size(layout.layout_id, qr_width, qr_height)
                    logger.info(
                        f"QR code detectado sem configuração, salvo no cache: ({qr_width}, {qr_height})"
                    )
                except ValueError as e:
                    logger.warning(f"Erro ao salvar no cache: {e}")
    
    # Se QR code foi detectado mas não foi processado acima, atualiza cache
    if anchor_type == "qrcode" and anchor_metadata and scaling_strategy == "none":
        qr_width = anchor_metadata.get("qr_width")
        qr_height = anchor_metadata.get("qr_height")
        if qr_width and qr_height:
            try:
                set_cached_qr_size(layout.layout_id, qr_width, qr_height)
                logger.debug(f"QR code salvo no cache: ({qr_width}, {qr_height})")
            except ValueError:
                pass  # Ignora erros silenciosamente neste caso
    
    # Extrai ROIs
    fields: list[Any] = []
    try:
        if anchor_position:
            fields = extract_all_rois(
                processed_image, anchor_position, layout, ocr_instance, scale_factor
            )
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

