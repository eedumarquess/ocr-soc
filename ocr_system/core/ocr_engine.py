"""Wrapper funcional do PaddleOCR."""

import logging
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

# Garante que o ccache esteja no PATH antes de importar o PaddleOCR
# Isso resolve o problema no Windows onde o PATH pode não estar atualizado
_ccache_path = os.path.join(os.path.expanduser("~"), "AppData", "Local", "ccache", "bin")
if os.path.exists(_ccache_path) and _ccache_path not in os.environ.get("PATH", ""):
    current_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{current_path};{_ccache_path}" if current_path else _ccache_path

# Configura variável de ambiente para garantir que o PaddleX use o cache local
# Isso evita downloads desnecessários dos modelos
_paddlex_cache_dir = Path.home() / ".paddlex" / "official_models"
if _paddlex_cache_dir.exists():
    # Define o diretório de cache do PaddleX explicitamente
    os.environ.setdefault("PADDLEX_HOME", str(_paddlex_cache_dir.parent))

# Suprime aviso do ccache do PaddlePaddle (não é crítico)
# O aviso aparece durante a compilação de extensões C++, mas não afeta a funcionalidade
warnings.filterwarnings(
    "ignore",
    message=".*ccache.*",
    category=UserWarning,
)

from paddleocr import PaddleOCR

from ocr_system.models.roi import OCRConfig
from ocr_system.utils.exceptions import OCRExtractionError
from ocr_system.utils.image_io import validate_image
from ocr_system.utils.medical_rules import clean_ocr_artifacts, normalize_medical_text, standardize_decimal


_ocr_instance: PaddleOCR | None = None


def initialize_ocr(lang: str = "pt", use_gpu: bool = True) -> PaddleOCR:
    """
    Inicializa instância única do PaddleOCR (singleton funcional).

    Args:
        lang: Idioma para OCR
        use_gpu: Usar GPU se disponível

    Returns:
        Instância do PaddleOCR
    """
    global _ocr_instance

    if _ocr_instance is None:
        # Configura o PaddleOCR para usar cache de modelos e evitar downloads desnecessários
        # use_textline_orientation=True: usa classificação de orientação de linha de texto
        # Os modelos são carregados do cache local em ~/.paddlex/official_models
        _ocr_instance = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
        )

    return _ocr_instance


def prepare_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Prepara imagem para processamento pelo PaddleOCR.

    Args:
        image: Imagem de entrada

    Returns:
        Imagem preparada em formato RGB uint8

    Raises:
        OCRExtractionError: Se a imagem não puder ser preparada
    """
    # Valida dimensões mínimas
    if len(image.shape) < 2 or image.shape[0] < 10 or image.shape[1] < 10:
        raise OCRExtractionError(f"Imagem muito pequena ou formato inválido: {image.shape}")

    # Copia a imagem para evitar modificar a original
    ocr_image = image.copy()

    # Remove NaN/Inf
    if np.any(np.isnan(ocr_image)) or np.any(np.isinf(ocr_image)):
        ocr_image = np.nan_to_num(ocr_image, nan=0.0, posinf=255.0, neginf=0.0)

    # Converte para uint8
    if ocr_image.dtype != np.uint8:
        if ocr_image.max() > 255 or ocr_image.min() < 0:
            if ocr_image.max() > ocr_image.min():
                ocr_image = ((ocr_image - ocr_image.min()) / (ocr_image.max() - ocr_image.min()) * 255).astype(np.uint8)
            else:
                ocr_image = np.zeros_like(ocr_image, dtype=np.uint8)
        else:
            ocr_image = ocr_image.astype(np.uint8)

    # Valida dimensões
    if len(ocr_image.shape) < 2:
        raise OCRExtractionError(f"Imagem com dimensões inválidas: {ocr_image.shape}")

    # Reduz dimensões extras se necessário
    if len(ocr_image.shape) > 3:
        ocr_image = ocr_image[:, :, :3]

    # Converte para RGB (3 canais)
    if len(ocr_image.shape) == 2:
        ocr_image = np.stack([ocr_image] * 3, axis=2)
    elif len(ocr_image.shape) == 3:
        if ocr_image.shape[2] == 1:
            ocr_image = np.repeat(ocr_image, 3, axis=2)
        elif ocr_image.shape[2] == 4:
            ocr_image = ocr_image[:, :, :3]
        elif ocr_image.shape[2] not in (3,):
            ocr_image = np.stack([ocr_image[:, :, 0]] * 3, axis=2)
    else:
        raise OCRExtractionError(f"Formato de imagem inválido: {ocr_image.shape}")

    # Valida formato final
    if len(ocr_image.shape) != 3 or ocr_image.shape[2] != 3:
        raise OCRExtractionError(f"Falha ao converter imagem para RGB: {ocr_image.shape}")

    if ocr_image.size == 0:
        raise OCRExtractionError("Imagem vazia após processamento")

    if ocr_image.shape[0] < 10 or ocr_image.shape[1] < 10:
        raise OCRExtractionError(f"Imagem muito pequena após processamento: {ocr_image.shape}")

    return ocr_image


def extract_from_ocr_result(first_page: Any) -> tuple[list[str], list[float], list[list[list[float]]]]:
    """
    Extrai texto, confiança e bounding boxes do resultado do OCR.

    Args:
        first_page: Primeira página do resultado do OCR (OCRResult ou lista)

    Returns:
        Tupla (texts, confidences, boxes)
    """
    texts: list[str] = []
    confidences: list[float] = []
    boxes: list[list[list[float]]] = []

    # Formato novo: objeto OCRResult (dict-like)
    if hasattr(first_page, 'keys') or hasattr(first_page, 'get'):
        keys = list(first_page.keys()) if hasattr(first_page, 'keys') else []
        
        # Extrai textos
        if 'rec_texts' in keys:
            value = first_page.get('rec_texts') if hasattr(first_page, 'get') else first_page['rec_texts']
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple)):
                    for t in value:
                        if t is not None:
                            if isinstance(t, np.ndarray):
                                t = t.tolist()
                            if isinstance(t, str) and t.strip():
                                texts.append(t)
                            elif isinstance(t, (list, tuple)) and len(t) > 0:
                                texts.append(' '.join(str(x) for x in t))
                elif isinstance(value, str):
                    texts.append(value)

        # Extrai confianças
        if 'rec_scores' in keys:
            value = first_page.get('rec_scores') if hasattr(first_page, 'get') else first_page['rec_scores']
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple)):
                    for c in value:
                        if c is not None:
                            if isinstance(c, np.ndarray):
                                c = float(c.item()) if c.size == 1 else float(c[0])
                            try:
                                confidences.append(float(c))
                            except (TypeError, ValueError):
                                continue
                elif isinstance(value, (int, float, np.number)):
                    confidences.append(float(value))

        # Extrai bounding boxes
        if 'dt_polys' in keys:
            value = first_page.get('dt_polys') if hasattr(first_page, 'get') else first_page['dt_polys']
            if value is not None:
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (list, tuple)):
                    for b in value:
                        if b is not None:
                            if isinstance(b, np.ndarray):
                                b = b.tolist()
                            if isinstance(b, (list, tuple)):
                                box_points = []
                                for p in b:
                                    if p is not None and isinstance(p, (list, tuple)) and len(p) >= 2:
                                        try:
                                            box_points.append([float(p[0]), float(p[1])])
                                        except (IndexError, TypeError, ValueError):
                                            continue
                                if box_points:
                                    boxes.append(box_points)

    # Formato antigo: lista de listas [[box, (text, confidence)], ...]
    elif isinstance(first_page, (list, tuple)):
        first_page = [item for item in first_page if item is not None]
        
        for line in first_page:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue

            box = line[0] if len(line) > 0 else None
            text_data = line[1] if len(line) > 1 else None

            if text_data is None:
                continue

            text = ""
            confidence = 0.0

            if isinstance(text_data, (list, tuple)):
                if len(text_data) >= 1:
                    text = str(text_data[0]) if text_data[0] is not None else ""
                if len(text_data) >= 2 and text_data[1] is not None:
                    try:
                        confidence = float(text_data[1])
                    except (TypeError, ValueError):
                        confidence = 0.0
            elif isinstance(text_data, str):
                text = text_data

            if text or confidence > 0:
                texts.append(text)
                confidences.append(confidence)
                if box and isinstance(box, (list, tuple)):
                    box_points = []
                    for p in box:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            try:
                                box_points.append([float(p[0]), float(p[1])])
                            except (IndexError, TypeError, ValueError):
                                continue
                    if box_points:
                        boxes.append(box_points)

    return texts, confidences, boxes


def extract_text_from_roi(
    image: np.ndarray, roi_config: OCRConfig, ocr_instance: PaddleOCR | None = None
) -> tuple[str, float, dict[str, Any]]:
    """
    Extrai texto de um ROI específico.

    Args:
        image: Imagem completa ou ROI já cortado
        roi_config: Configuração de OCR do ROI
        ocr_instance: Instância do PaddleOCR (opcional, cria se None)

    Returns:
        Tupla (texto extraído, confiança média, metadados)

    Raises:
        OCRExtractionError: Se a extração falhar
    """
    validate_image(image)

    # Prepara imagem
    try:
        ocr_image = prepare_image_for_ocr(image)
    except OCRExtractionError as e:
        return "", 0.0, {"error": str(e)}
    except Exception as e:
        return "", 0.0, {"error": f"Erro ao preparar imagem: {str(e)}"}

    # Inicializa OCR se necessário
    if ocr_instance is None:
        ocr_instance = initialize_ocr(lang=roi_config.lang)

    # Executa OCR
    try:
        result = ocr_instance.ocr(ocr_image)
    except Exception as e:
        return "", 0.0, {"error": f"Erro ao executar OCR: {str(e)}"}

    # Valida resultado
    if result is None:
        return "", 0.0, {"error": "Resultado do OCR é None"}

    if not isinstance(result, (list, tuple)) or len(result) == 0:
        return "", 0.0, {"error": "Resultado do OCR está vazio"}

    # Acessa primeira página
    try:
        first_page = result[0]
    except (IndexError, TypeError) as e:
        return "", 0.0, {"error": f"Não foi possível acessar primeira página: {str(e)}"}

    if first_page is None:
        return "", 0.0, {"error": "Nenhum texto detectado"}

    # Extrai dados usando função auxiliar
    try:
        texts, confidences, boxes = extract_from_ocr_result(first_page)
    except Exception as e:
        return "", 0.0, {"error": f"Erro ao extrair dados do OCR: {str(e)}"}

    if not texts:
        return "", 0.0, {"error": "Nenhum texto extraído"}

    # Combina textos e calcula confiança média
    raw_text = " ".join(texts)
    avg_confidence = float(sum(confidences) / len(confidences) if confidences else 0.0)

    # Aplica pós-processamento
    processed_text = postprocess_text(raw_text, roi_config)

    # Prepara metadados
    metadata: dict[str, Any] = {
        "raw_text": raw_text,
        "confidence_scores": [float(c) for c in confidences],
        "bounding_boxes": boxes,
        "num_detections": int(len(texts)),
    }

    # Log apenas em modo debug (não mais INFO)
    # Os resultados serão formatados em JSON pela função format_output_json

    return processed_text, avg_confidence, metadata


def postprocess_text(text: str, config: OCRConfig) -> str:
    """
    Aplica pós-processamento ao texto extraído.

    Args:
        text: Texto bruto
        config: Configuração de OCR

    Returns:
        Texto processado
    """
    if not text:
        return text

    # Remove artefatos
    text = clean_ocr_artifacts(text)

    # Aplica transformações de preprocessamento
    for transform in config.preprocessing:
        if transform == "uppercase":
            text = text.upper()
        elif transform == "lowercase":
            text = text.lower()
        elif transform == "strip":
            text = text.strip()
        elif transform == "remove_spaces":
            text = text.replace(" ", "")

    # Filtra caracteres permitidos se especificado
    if config.allowed_chars:
        import re

        pattern = f"[{config.allowed_chars}]"
        text = "".join(re.findall(pattern, text))

    # Normaliza decimais para numéricos
    if config.type == "numeric" and config.decimal_separator:
        text = standardize_decimal(text, config.decimal_separator)

    # Remove unidade se presente e esperada
    if config.expected_unit and config.expected_unit in text:
        text = text.replace(config.expected_unit, "").strip()

    return text

