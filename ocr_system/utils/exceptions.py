"""Exceções customizadas do sistema."""


class InvalidImageError(Exception):
    """Erro quando imagem é inválida ou não pode ser carregada."""

    pass


class AnchorNotFoundError(Exception):
    """Erro quando âncora não pode ser detectada."""

    pass


class ValidationError(Exception):
    """Erro quando validação de campo falha."""

    pass


class OCRExtractionError(Exception):
    """Erro durante extração de OCR."""

    pass


