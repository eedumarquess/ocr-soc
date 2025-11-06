"""Detecção de âncoras (QR Code e Texto)."""

import re
from typing import Any

import cv2
import numpy as np
from pyzbar import pyzbar
from rapidfuzz import fuzz

from ocr_system.core.ocr_engine import extract_text_from_roi, initialize_ocr
from ocr_system.models.anchor import AnchorStrategy, QRCodeAnchorConfig, TextAnchorConfig
from ocr_system.models.roi import OCRConfig
from ocr_system.utils.exceptions import AnchorNotFoundError
from ocr_system.utils.geometry import crop_roi, get_center
from ocr_system.utils.image_io import validate_image, image_to_grayscale


def _preprocess_for_qrcode(image: np.ndarray) -> list[tuple[np.ndarray, str]]:
    """
    Prepara múltiplas versões da imagem para detecção de QR code.
    
    Args:
        image: Imagem original
        
    Returns:
        Lista de tuplas (imagem processada, descrição)
    """
    variants = []
    
    # 1. Original em escala de cinza
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image
    variants.append((gray, "grayscale"))
    
    # 2. Binarização Otsu (melhor para QR codes)
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append((binary_otsu, "otsu_binary"))
    
    # 3. Binarização adaptativa (mais suave)
    block_size = 11
    if block_size % 2 == 0:
        block_size += 1
    binary_adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
    )
    variants.append((binary_adaptive, "adaptive_binary"))
    
    # 4. Melhoria de contraste (CLAHE) + escala de cinza
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced_gray = image_to_grayscale(enhanced)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
    variants.append((enhanced_gray, "contrast_enhanced"))
    
    # 5. Inversão (caso o QR code seja branco em fundo preto)
    inverted = cv2.bitwise_not(gray)
    variants.append((inverted, "inverted"))
    
    return variants


def detect_qrcode(
    image: np.ndarray, config: QRCodeAnchorConfig
) -> tuple[tuple[int, int], float, dict[str, Any]] | None:
    """
    Detecta QR Code na imagem usando múltiplas estratégias de pré-processamento.

    Args:
        image: Imagem para buscar
        config: Configuração de detecção de QR Code

    Returns:
        Tupla (posição (x, y), confiança, metadados) ou None se não encontrado
    """
    validate_image(image)

    # Aplica região de busca se especificada
    search_image = image
    offset_x, offset_y = 0, 0

    if config.search_region:
        x, y, width, height = config.search_region
        offset_x, offset_y = x, y
        search_image = crop_roi(image, (x, y, width, height))

    # Tenta múltiplas versões pré-processadas da imagem
    variants = _preprocess_for_qrcode(search_image)
    
    for variant_image, variant_name in variants:
        try:
            # Detecta QR codes
            qr_codes = pyzbar.decode(variant_image)

            if not qr_codes:
                continue

            # Processa cada QR code encontrado
            for qr in qr_codes:
                try:
                    content = qr.data.decode("utf-8")
                except Exception:
                    # Se não conseguir decodificar, tenta próximo
                    continue
                    
                rect = qr.rect

                # Valida conteúdo se pattern fornecido
                if config.expected_content_pattern:
                    if not re.match(config.expected_content_pattern, content):
                        continue

                # Calcula posição absoluta (centro do QR)
                center_x = offset_x + rect.left + rect.width // 2
                center_y = offset_y + rect.top + rect.height // 2

                # Confiança padrão (pyzbar não retorna confiança, usa 1.0 se encontrado)
                confidence = float(1.0)

                metadata = {
                    "content": content,
                    "rect": (int(rect.left), int(rect.top), int(rect.width), int(rect.height)),
                    "type": "qrcode",
                    "preprocessing_variant": variant_name,
                }

                if confidence >= config.min_confidence:
                    return ((int(center_x), int(center_y)), confidence, metadata)
        except Exception:
            # Continua tentando outras variantes
            continue

    return None


def detect_text_anchor(
    image: np.ndarray, config: TextAnchorConfig, ocr_instance: Any = None
) -> tuple[tuple[int, int], float, dict[str, Any]] | None:
    """
    Detecta âncora de texto na imagem.

    Args:
        image: Imagem para buscar
        config: Configuração de detecção de texto
        ocr_instance: Instância do OCR (opcional)

    Returns:
        Tupla (posição (x, y), confiança, metadados) ou None se não encontrado
    """
    validate_image(image)

    # Aplica região de busca se especificada
    search_image = image
    offset_x, offset_y = 0, 0

    if config.search_region:
        x, y, width, height = config.search_region
        offset_x, offset_y = x, y
        search_image = crop_roi(image, (x, y, width, height))

    # Configura OCR para busca rápida
    ocr_config = OCRConfig(type="text", lang="pt")
    if ocr_instance is None:
        ocr_instance = initialize_ocr(lang="pt")

    try:
        # Executa OCR na região
        text, confidence, metadata = extract_text_from_roi(search_image, ocr_config, ocr_instance)

        if not text:
            return None

        # Busca keyword com fuzzy matching se habilitado
        if config.fuzzy_match:
            # Compara com cada palavra/palavras do texto
            words = text.split()
            best_match = None
            best_score = 0.0
            best_position = None

            for i, word in enumerate(words):
                score = fuzz.ratio(config.keyword.upper(), word.upper()) / 100.0
                if score > best_score and score >= (1.0 - config.max_distance / 10.0):
                    best_score = score
                    best_match = word
                    # Tenta encontrar posição da palavra nas bounding boxes
                    if "bounding_boxes" in metadata and i < len(metadata["bounding_boxes"]):
                        box = metadata["bounding_boxes"][i]
                        best_position = get_center((box[0][0], box[0][1], box[2][0] - box[0][0], box[2][1] - box[0][1]))

            if best_match and best_position:
                abs_x = int(offset_x + best_position[0])
                abs_y = int(offset_y + best_position[1])
                return ((abs_x, abs_y), float(best_score), {"matched_word": best_match, "type": "text"})
        else:
            # Busca exata
            if config.keyword.upper() in text.upper():
                # Usa primeira ocorrência - centro aproximado da região
                if "bounding_boxes" in metadata and len(metadata["bounding_boxes"]) > 0:
                    # Tenta encontrar box que contém a keyword
                    for box in metadata["bounding_boxes"]:
                        center = get_center((box[0][0], box[0][1], box[2][0] - box[0][0], box[2][1] - box[0][1]))
                        abs_x = int(offset_x + center[0])
                        abs_y = int(offset_y + center[1])
                        return ((abs_x, abs_y), float(confidence), {"type": "text"})

                # Fallback: centro da região de busca
                if config.search_region:
                    x, y, w, h = config.search_region
                    return ((int(x + w // 2), int(y + h // 2)), float(confidence), {"type": "text"})
                else:
                    h, w = image.shape[:2]
                    return ((int(w // 2), int(h // 2)), float(confidence), {"type": "text"})

    except Exception:
        return None

    return None


def detect_anchor(
    image: np.ndarray, strategy: AnchorStrategy, ocr_instance: Any = None
) -> tuple[tuple[int, int], float, str, dict[str, Any]]:
    """
    Detecta âncora usando estratégia com fallbacks.

    Args:
        image: Imagem para processar
        strategy: Estratégia de detecção
        ocr_instance: Instância do OCR (opcional)

    Returns:
        Tupla (posição (x, y), confiança, tipo, metadados)

    Raises:
        AnchorNotFoundError: Se nenhuma âncora for detectada
    """
    validate_image(image)

    # Tenta âncora primária
    result = None
    anchor_type = None

    if isinstance(strategy.primary, QRCodeAnchorConfig):
        result = detect_qrcode(image, strategy.primary)
        anchor_type = "qrcode"
    elif isinstance(strategy.primary, TextAnchorConfig):
        result = detect_text_anchor(image, strategy.primary, ocr_instance)
        anchor_type = "text"

    # Verifica confiança da primária
    if result:
        position, confidence, metadata = result
        if confidence >= strategy.min_confidence_threshold:
            return (position, confidence, anchor_type, metadata)
        elif not strategy.fallback_on_low_confidence:
            # Aceita mesmo com baixa confiança se fallback desabilitado
            return (position, confidence, anchor_type, metadata)

    # Tenta fallbacks
    for fallback in strategy.fallbacks:
        if isinstance(fallback, QRCodeAnchorConfig):
            result = detect_qrcode(image, fallback)
            anchor_type = "qrcode"
        elif isinstance(fallback, TextAnchorConfig):
            result = detect_text_anchor(image, fallback, ocr_instance)
            anchor_type = "text"

        if result:
            position, confidence, metadata = result
            if confidence >= strategy.min_confidence_threshold:
                return (position, confidence, anchor_type, metadata)

    # Nenhuma âncora encontrada
    raise AnchorNotFoundError(
        f"Nenhuma âncora detectada. Tentadas: {strategy.primary.type} + {len(strategy.fallbacks)} fallbacks"
    )


def detect_anchor_with_fallback(
    processed_image: np.ndarray,
    original_image: np.ndarray,
    strategy: AnchorStrategy,
    ocr_instance: Any = None,
) -> tuple[tuple[int, int], float, str, dict[str, Any]]:
    """
    Detecta âncora com fallback para imagem original se configurado.
    
    Esta função tenta detectar a âncora primeiro na imagem processada.
    Se falhar e `strategy.try_original_on_failure` for True, tenta na imagem original.
    
    Args:
        processed_image: Imagem após pré-processamento
        original_image: Imagem original (sem pré-processamento)
        strategy: Estratégia de detecção de âncora
        ocr_instance: Instância do OCR (opcional)
    
    Returns:
        Tupla (posição (x, y), confiança, tipo, metadados)
    
    Raises:
        AnchorNotFoundError: Se nenhuma âncora for detectada em nenhuma das imagens
    """
    validate_image(processed_image)
    validate_image(original_image)
    
    # Tenta detectar na imagem processada primeiro
    try:
        position, confidence, a_type, metadata = detect_anchor(
            processed_image, strategy, ocr_instance
        )
        # Adiciona flag indicando que foi detectado na imagem processada
        if metadata is None:
            metadata = {}
        metadata["detected_on_processed"] = True
        return (position, confidence, a_type, metadata)
    except AnchorNotFoundError as e:
        # Se falhar na imagem processada e fallback estiver habilitado, tenta na original
        if strategy.try_original_on_failure:
            try:
                position, confidence, a_type, metadata = detect_anchor(
                    original_image, strategy, ocr_instance
                )
                # Adiciona flag indicando que foi detectado na imagem original
                if metadata is None:
                    metadata = {}
                metadata["detected_on_original"] = True
                return (position, confidence, a_type, metadata)
            except AnchorNotFoundError as e2:
                # Ambas falharam, levanta erro com informações de ambas as tentativas
                raise AnchorNotFoundError(
                    f"Âncora não detectada em nenhuma imagem. "
                    f"Processada: {str(e)}. Original: {str(e2)}"
                ) from e2
        else:
            # Fallback desabilitado, apenas propaga o erro original
            raise

