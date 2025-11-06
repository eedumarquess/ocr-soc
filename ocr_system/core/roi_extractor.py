"""Extração de ROIs com coordenadas relativas."""

import logging
import re
from typing import Any

import numpy as np

from ocr_system.core.ocr_engine import extract_text_from_roi, initialize_ocr
from ocr_system.models.layout import LayoutConfig
from ocr_system.models.result import FieldResult
from ocr_system.models.roi import ROIConfig
from ocr_system.utils.exceptions import ValidationError
from ocr_system.utils.geometry import calculate_absolute_roi, calculate_scaled_roi, crop_roi, is_roi_valid
from ocr_system.utils.image_io import get_image_shape, validate_image
from ocr_system.utils.medical_rules import normalize_medical_text, remove_label_from_text


def validate_field(value: str | None, roi_config: ROIConfig) -> tuple[bool, str | None]:
    """
    Valida campo extraído conforme configuração.

    Args:
        value: Valor a validar
        roi_config: Configuração do ROI

    Returns:
        Tupla (válido, mensagem de erro)
    """
    validation = roi_config.validation

    # Campo obrigatório vazio
    if validation.required and (not value or not value.strip()):
        return False, "Campo obrigatório está vazio"

    if not value or not value.strip():
        return True, None  # Campo opcional vazio é válido

    # Validação regex
    if validation.regex:
        if not re.match(validation.regex, value):
            return False, f"Valor não corresponde ao padrão: {validation.regex}"

    # Validação de range (para numéricos)
    if validation.range and roi_config.ocr_config.type == "numeric":
        try:
            # Remove separador decimal e converte
            num_value = float(value.replace(",", "."))
            min_val, max_val = validation.range
            if not (min_val <= num_value <= max_val):
                return False, f"Valor {num_value} fora do range [{min_val}, {max_val}]"
        except ValueError:
            return False, f"Não foi possível converter '{value}' para número"

    return True, None


def extract_roi_field(
    image: np.ndarray,
    anchor_position: tuple[int, int],
    roi_config: ROIConfig,
    ocr_instance: Any = None,
    scale_factor: float | tuple[float, float] = 1.0,
) -> FieldResult:
    """
    Extrai e valida um campo de um ROI.

    Args:
        image: Imagem processada
        anchor_position: Posição da âncora (x, y)
        roi_config: Configuração do ROI
        ocr_instance: Instância do OCR (opcional)
        scale_factor: Fator de escala único (float) ou tupla (scale_x, scale_y) para escalas separadas

    Returns:
        Resultado da extração do campo
    """
    validate_image(image)
    image_shape = get_image_shape(image)

    # Aplica escala se necessário
    if scale_factor != 1.0:
        scaled_position, scaled_size = calculate_scaled_roi(
            roi_config.relative_position, roi_config.size, scale_factor
        )
    else:
        scaled_position = roi_config.relative_position
        scaled_size = roi_config.size

    # Calcula ROI absoluto
    absolute_roi = calculate_absolute_roi(
        anchor_position, scaled_position, scaled_size, image_shape
    )

    # Valida ROI
    if not is_roi_valid(absolute_roi, image_shape):
        return FieldResult(
            roi_id=roi_config.id,
            value=None,
            confidence=0.0,
            validated=False,
            validation_error=f"ROI inválido: {absolute_roi}",
            metadata={"required": roi_config.validation.required},
        )

    # Extrai região
    roi_image = crop_roi(image, absolute_roi)

    # Executa OCR
    if ocr_instance is None:
        ocr_instance = initialize_ocr(lang=roi_config.ocr_config.lang)

    try:
        text, confidence, ocr_metadata = extract_text_from_roi(roi_image, roi_config.ocr_config, ocr_instance)
        raw_text = ocr_metadata.get("raw_text", text)
    except Exception as e:
        return FieldResult(
            roi_id=roi_config.id,
            value=None,
            confidence=0.0,
            raw_text=None,
            validated=False,
            validation_error=f"Erro no OCR: {str(e)}",
            metadata={"required": roi_config.validation.required},
        )

    # Remove label do texto extraído
    processed_value = remove_label_from_text(text, roi_config.label)

    # Aplica normalização médica se configurado
    # (será aplicado no post-processing, mas podemos aplicar aqui também)
    # processed_value já tem a label removida

    # Valida campo
    is_valid, error_msg = validate_field(processed_value, roi_config)

    # Converte valores numpy para tipos Python nativos
    confidence_float = float(confidence) if confidence is not None else None
    
    # Limpa metadados OCR de tipos numpy
    ocr_metadata_clean = {}
    if ocr_metadata:
        for key, value in ocr_metadata.items():
            if isinstance(value, (list, tuple)):
                ocr_metadata_clean[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
            elif isinstance(value, (np.floating, np.integer)):
                ocr_metadata_clean[key] = float(value)
            else:
                ocr_metadata_clean[key] = value

    # Log apenas erros de validação (warnings)
    if error_msg:
        logger = logging.getLogger(__name__)
        logger.warning(f"Campo {roi_config.id}: Erro de validação - {error_msg}")

    return FieldResult(
        roi_id=roi_config.id,
        value=processed_value if processed_value else None,
        confidence=confidence_float,
        raw_text=raw_text,
        validated=is_valid,
        validation_error=error_msg,
        metadata={
            "required": roi_config.validation.required,
            "absolute_roi": tuple(int(x) for x in absolute_roi),
            "ocr_metadata": ocr_metadata_clean,
        },
    )


def extract_all_rois(
    image: np.ndarray,
    anchor_position: tuple[int, int],
    layout: LayoutConfig,
    ocr_instance: Any = None,
    scale_factor: float | tuple[float, float] = 1.0,
) -> list[FieldResult]:
    """
    Extrai todos os ROIs do layout.

    Args:
        image: Imagem processada
        anchor_position: Posição da âncora (x, y)
        layout: Configuração do layout
        ocr_instance: Instância do OCR (opcional)
        scale_factor: Fator de escala único (float) ou tupla (scale_x, scale_y) para escalas separadas

    Returns:
        Lista de resultados dos campos
    """
    validate_image(image)

    if ocr_instance is None:
        ocr_instance = initialize_ocr()

    results = []

    for roi_config in layout.rois:
        field_result = extract_roi_field(
            image, anchor_position, roi_config, ocr_instance, scale_factor
        )
        results.append(field_result)

    return results

