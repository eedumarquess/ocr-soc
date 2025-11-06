"""Cálculos geométricos para coordenadas e ROIs."""

from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from ocr_system.models.anchor import QRCodeAnchorConfig

from ocr_system.utils.qr_cache import get_cached_qr_size


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

    # Calcula posição absoluta com precisão melhorada
    # Usa arredondamento explícito para evitar erros de precisão
    abs_x = round(anchor_x + offset_x)
    abs_y = round(anchor_y + offset_y)
    abs_x = int(abs_x)
    abs_y = int(abs_y)

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


def calculate_scaled_roi(
    relative_position: tuple[int, int],
    roi_size: tuple[int, int],
    scale_factor: float | tuple[float, float],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Aplica fator de escala à posição relativa e tamanho do ROI.

    Usado para redimensionar ROIs quando o QR code detectado tem tamanho
    diferente do esperado, mantendo as proporções relativas.

    Args:
        relative_position: Posição relativa à âncora (offset_x, offset_y)
        roi_size: Tamanho do ROI (width, height)
        scale_factor: Fator de escala único (float) ou tupla (scale_x, scale_y) para escalas separadas

    Returns:
        Tupla ((scaled_offset_x, scaled_offset_y), (scaled_width, scaled_height))
    """
    offset_x, offset_y = relative_position
    width, height = roi_size

    # Suporta escala única ou escalas separadas para X e Y
    if isinstance(scale_factor, tuple):
        scale_x, scale_y = scale_factor
    else:
        scale_x = scale_y = scale_factor

    # Aplica escalas separadas para X e Y com maior precisão
    # Usa arredondamento com mais casas decimais para evitar erros acumulados
    # Multiplica primeiro, depois arredonda com precisão
    scaled_offset_x = round(offset_x * scale_x, 2)  # 2 casas decimais antes de int
    scaled_offset_y = round(offset_y * scale_y, 2)
    scaled_width = round(width * scale_x, 2)
    scaled_height = round(height * scale_y, 2)
    
    # Converte para int após arredondamento preciso
    scaled_offset_x = int(scaled_offset_x)
    scaled_offset_y = int(scaled_offset_y)
    scaled_width = int(scaled_width)
    scaled_height = int(scaled_height)

    # Garante tamanhos mínimos
    scaled_width = max(1, scaled_width)
    scaled_height = max(1, scaled_height)

    return (
        (scaled_offset_x, scaled_offset_y),
        (scaled_width, scaled_height),
    )


def calculate_expected_qr_size(
    image: np.ndarray,
    config: "QRCodeAnchorConfig",
    layout_id: str,
) -> tuple[int, int] | None:
    """
    Calcula tamanho esperado do QR code usando estratégias híbridas.

    Ordem de prioridade:
    1. Proporções relativas (expected_qr_width_ratio e expected_qr_height_ratio)
    2. Tamanho fixo (expected_qr_size)
    3. Cache por layout
    4. None (sem referência - usar QR detectado como referência)

    Args:
        image: Imagem para calcular proporções
        config: Configuração da âncora QR code
        layout_id: ID do layout para consulta de cache

    Returns:
        Tupla (width, height) do tamanho esperado ou None se nenhuma estratégia disponível
    """
    img_height, img_width = image.shape[:2]

    # Estratégia 1: Proporções relativas
    if config.expected_qr_width_ratio is not None and config.expected_qr_height_ratio is not None:
        expected_width = int(img_width * config.expected_qr_width_ratio)
        expected_height = int(img_height * config.expected_qr_height_ratio)
        # Valida que os tamanhos calculados são positivos
        if expected_width > 0 and expected_height > 0:
            return (expected_width, expected_height)

    # Estratégia 2: Tamanho fixo
    if config.expected_qr_size is not None:
        return config.expected_qr_size

    # Estratégia 3: Cache
    cached = get_cached_qr_size(layout_id)
    if cached is not None:
        return cached

    # Estratégia 4: Sem referência (None = usar QR detectado como referência)
    return None

