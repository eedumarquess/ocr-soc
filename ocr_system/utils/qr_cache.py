"""Cache de tamanhos de QR code por layout."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Cache em memória: layout_id -> (width, height)
_layout_qr_cache: dict[str, tuple[int, int]] = {}


def get_cached_qr_size(layout_id: str) -> tuple[int, int] | None:
    """
    Retorna tamanho de QR code em cache para o layout.

    Args:
        layout_id: ID do layout

    Returns:
        Tupla (width, height) se encontrado no cache, None caso contrário
    """
    return _layout_qr_cache.get(layout_id)


def set_cached_qr_size(layout_id: str, width: int, height: int) -> None:
    """
    Salva tamanho de QR code no cache.

    Args:
        layout_id: ID do layout
        width: Largura do QR code detectado
        height: Altura do QR code detectado

    Raises:
        ValueError: Se width ou height forem <= 0
    """
    if width <= 0 or height <= 0:
        error_msg = f"Tamanhos inválidos: width={width}, height={height}. Devem ser > 0"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    _layout_qr_cache[layout_id] = (width, height)
    logger.debug(f"Tamanho de QR code salvo no cache para layout '{layout_id}': ({width}, {height})")


def clear_cache(layout_id: str | None = None) -> None:
    """
    Limpa cache (todo ou de um layout específico).

    Args:
        layout_id: ID do layout para limpar. Se None, limpa todo o cache.
    """
    if layout_id:
        removed = _layout_qr_cache.pop(layout_id, None)
        if removed:
            logger.debug(f"Cache limpo para layout '{layout_id}'")
        else:
            logger.debug(f"Layout '{layout_id}' não encontrado no cache")
    else:
        count = len(_layout_qr_cache)
        _layout_qr_cache.clear()
        logger.debug(f"Todo o cache foi limpo ({count} entradas removidas)")

