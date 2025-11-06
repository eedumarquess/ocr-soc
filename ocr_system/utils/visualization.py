"""Utilitários para visualização de ROIs e resultados."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr_system.models.layout import LayoutConfig
from ocr_system.models.result import FieldResult, OCRResult
from ocr_system.utils.geometry import calculate_absolute_roi
from ocr_system.utils.image_io import get_image_shape


def draw_rois_on_image(
    image: np.ndarray,
    layout: LayoutConfig,
    anchor_position: tuple[int, int] | None,
    field_results: list[FieldResult],
    show_labels: bool = True,
    show_values: bool = False,
) -> np.ndarray:
    """
    Desenha ROIs na imagem com cores e labels.

    Args:
        image: Imagem onde desenhar
        layout: Configuração do layout
        anchor_position: Posição da âncora (x, y) ou None
        field_results: Resultados dos campos extraídos
        show_labels: Se deve mostrar labels dos ROIs
        show_values: Se deve mostrar valores extraídos

    Returns:
        Imagem com ROIs desenhados
    """
    # Converter para BGR se necessário (OpenCV usa BGR)
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()

    image_shape = get_image_shape(image)

    # Cores por tipo de OCR
    colors = {
        "text": (0, 255, 255),  # Amarelo (BGR)
        "numeric": (255, 255, 0),  # Ciano (BGR)
        "date": (255, 0, 255),  # Magenta (BGR)
    }

    # Desenhar âncora se disponível
    if anchor_position:
        anchor_x, anchor_y = anchor_position
        cv2.circle(vis_image, (anchor_x, anchor_y), 15, (0, 255, 255), 3)
        cv2.line(vis_image, (anchor_x - 20, anchor_y), (anchor_x + 20, anchor_y), (0, 255, 255), 2)
        cv2.line(vis_image, (anchor_x, anchor_y - 20), (anchor_x, anchor_y + 20), (0, 255, 255), 2)
        cv2.putText(
            vis_image,
            "ANCORA",
            (anchor_x + 25, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    # Criar mapa de resultados por ROI ID
    results_map = {result.roi_id: result for result in field_results}

    # Desenhar cada ROI
    for i, roi_config in enumerate(layout.rois):
        if not anchor_position:
            continue

        # Calcular ROI absoluto
        absolute_roi = calculate_absolute_roi(
            anchor_position, roi_config.relative_position, roi_config.size, image_shape
        )

        x, y, w, h = absolute_roi

        # Obter cor baseada no tipo
        color = colors.get(roi_config.ocr_config.type, (255, 255, 255))

        # Obter resultado se disponível
        result = results_map.get(roi_config.id)
        is_valid = result.validated if result else False
        has_value = result.value is not None and result.value.strip() != "" if result else False

        # Ajustar cor baseado no status
        if not has_value:
            color = (128, 128, 128)  # Cinza se vazio
        elif not is_valid:
            color = (0, 0, 255)  # Vermelho se inválido

        # Desenhar retângulo
        thickness = 3 if has_value and is_valid else 2
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)

        # Desenhar label
        if show_labels:
            label = f"{i+1}: {roi_config.label[:30]}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(15, y - 5)

            # Fundo preto para legibilidade
            cv2.rectangle(
                vis_image,
                (x, label_y - label_size[1] - 5),
                (x + label_size[0] + 5, label_y + 5),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                vis_image,
                label,
                (x + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # Desenhar valor extraído se solicitado
        if show_values and result and result.value:
            value_text = f"{result.value[:40]}"
            value_size, _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            value_y = y + h + value_size[1] + 5

            # Fundo preto para legibilidade
            cv2.rectangle(
                vis_image,
                (x, value_y - value_size[1] - 2),
                (x + value_size[0] + 4, value_y + 2),
                (0, 0, 0),
                -1,
            )
            cv2.putText(
                vis_image,
                value_text,
                (x + 2, value_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

            # Mostrar confiança se disponível
            if result.confidence is not None:
                conf_text = f"Conf: {result.confidence:.2f}"
                conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                conf_y = value_y + conf_size[1] + 5
                cv2.rectangle(
                    vis_image,
                    (x, conf_y - conf_size[1] - 2),
                    (x + conf_size[0] + 4, conf_y + 2),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    vis_image,
                    conf_text,
                    (x + 2, conf_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (200, 200, 200),
                    1,
                )

    return vis_image


def save_roi_visualization(
    image: np.ndarray,
    result: OCRResult,
    layout: LayoutConfig,
    output_path: Path,
    show_values: bool = True,
) -> None:
    """
    Salva imagem com ROIs desenhados.

    Args:
        image: Imagem processada
        result: Resultado do processamento
        layout: Configuração do layout
        output_path: Caminho para salvar a imagem
        show_values: Se deve mostrar valores extraídos
    """
    vis_image = draw_rois_on_image(
        image,
        layout,
        result.anchor_position,
        result.fields,
        show_labels=True,
        show_values=show_values,
    )

    cv2.imwrite(str(output_path), vis_image)

