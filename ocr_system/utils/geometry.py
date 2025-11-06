"""Cálculos geométricos para coordenadas e ROIs."""

from typing import Tuple

import numpy as np


def calculate_absolute_roi(
    anchor_position: tuple[int, int],
    relative_position: tuple[int, int],
    roi_size: tuple[int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Calcula coordenadas absolutas do ROI a partir da posição relativa à âncora.

    Args:
        anchor_position: Posição da âncora (x, y)
        relative_position: Offset relativo à âncora (offset_x, offset_y)
        roi_size: Tamanho do ROI (width, height)
        image_shape: Dimensões da imagem (height, width)

    Returns:
        Tupla (x, y, width, height) do ROI absoluto
    """
    anchor_x, anchor_y = anchor_position
    offset_x, offset_y = relative_position
    width, height = roi_size
    img_height, img_width = image_shape

    # Calcula posição absoluta
    abs_x = anchor_x + offset_x
    abs_y = anchor_y + offset_y

    # Garante que o ROI não ultrapasse os limites da imagem
    abs_x = max(0, min(abs_x, img_width - 1))
    abs_y = max(0, min(abs_y, img_height - 1))

    # Ajusta tamanho se necessário
    max_width = img_width - abs_x
    max_height = img_height - abs_y
    width = min(width, max_width)
    height = min(height, max_height)

    return (abs_x, abs_y, width, height)


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """
    Extrai região de interesse da imagem.

    Args:
        image: Imagem completa
        roi: ROI (x, y, width, height)

    Returns:
        Imagem cortada do ROI
    """
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def get_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """
    Calcula centro de uma bounding box.

    Args:
        bbox: Bounding box (x, y, width, height)

    Returns:
        Centro (x, y)
    """
    x, y, width, height = bbox
    center_x = x + width // 2
    center_y = y + height // 2
    return (center_x, center_y)


def is_roi_valid(roi: tuple[int, int, int, int], image_shape: tuple[int, int]) -> bool:
    """
    Valida se ROI está dentro dos limites da imagem.

    Args:
        roi: ROI (x, y, width, height)
        image_shape: Dimensões da imagem (height, width)

    Returns:
        True se válido, False caso contrário
    """
    x, y, width, height = roi
    img_height, img_width = image_shape

    if x < 0 or y < 0:
        return False
    if x + width > img_width or y + height > img_height:
        return False
    if width <= 0 or height <= 0:
        return False

    return True


