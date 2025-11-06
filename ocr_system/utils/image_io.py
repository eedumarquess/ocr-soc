"""Operações de I/O de imagens."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ocr_system.utils.exceptions import InvalidImageError


def load_image(image_path: str | Path) -> np.ndarray:
    """
    Carrega imagem do disco.

    Args:
        image_path: Caminho para a imagem

    Returns:
        Imagem como array numpy (BGR para OpenCV)

    Raises:
        InvalidImageError: Se a imagem não puder ser carregada
    """
    path = Path(image_path)
    if not path.exists():
        raise InvalidImageError(f"Arquivo não encontrado: {image_path}")

    img = cv2.imread(str(path))
    if img is None:
        raise InvalidImageError(f"Não foi possível carregar imagem: {image_path}")

    if img.size == 0:
        raise InvalidImageError(f"Imagem vazia: {image_path}")

    return img


def save_image(image: np.ndarray, output_path: str | Path) -> None:
    """
    Salva imagem no disco.

    Args:
        image: Imagem como array numpy
        output_path: Caminho de saída
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def image_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Converte imagem BGR (OpenCV) para RGB.

    Args:
        image: Imagem BGR

    Returns:
        Imagem RGB
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def image_to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Converte imagem RGB para BGR (OpenCV).

    Args:
        image: Imagem RGB

    Returns:
        Imagem BGR
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def image_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converte imagem para escala de cinza.

    Args:
        image: Imagem colorida ou em escala de cinza

    Returns:
        Imagem em escala de cinza
    """
    if len(image.shape) == 2:
        return image
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def validate_image(image: np.ndarray) -> None:
    """
    Valida se a imagem é válida.

    Args:
        image: Imagem para validar

    Raises:
        InvalidImageError: Se a imagem for inválida
    """
    if image is None:
        raise InvalidImageError("Imagem é None")
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Imagem deve ser numpy array, recebido: {type(image)}")
    if image.size == 0:
        raise InvalidImageError("Imagem está vazia")
    if len(image.shape) < 2:
        raise InvalidImageError(f"Dimensões inválidas: {image.shape}")


def get_image_shape(image: np.ndarray) -> tuple[int, int]:
    """
    Retorna dimensões da imagem (height, width).

    Args:
        image: Imagem

    Returns:
        Tupla (height, width)
    """
    validate_image(image)
    if len(image.shape) == 2:
        return image.shape[0], image.shape[1]
    return image.shape[0], image.shape[1]

