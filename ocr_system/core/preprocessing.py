"""Pipeline de pré-processamento de imagens."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage import filters, restoration
from skimage.exposure import equalize_adapthist

from ocr_system.models.layout import PreprocessingConfig
from ocr_system.utils.exceptions import InvalidImageError
from ocr_system.utils.image_io import validate_image, image_to_grayscale


def deskew_image(image: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, float]:
    """
    Corrige rotação da imagem (deskew) focando em linhas pretas horizontais.

    Melhora a detecção de inclinação ao:
    - Binarizar a imagem para focar em linhas pretas (texto)
    - Usar morfologia para destacar linhas horizontais
    - Aplicar HoughLinesP para detecção mais robusta
    - Filtrar e ponderar linhas horizontais relevantes

    Args:
        image: Imagem em escala de cinza ou colorida
        threshold: Graus mínimos para aplicar correção

    Returns:
        Tupla (imagem corrigida, ângulo detectado)
    """
    validate_image(image)
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image
    h, w = gray.shape

    # 1. Binarizar para focar em linhas pretas (texto/documento)
    # Usa Otsu para melhor separação texto/fundo
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Morfologia para destacar linhas horizontais
    # Kernel horizontal longo para detectar linhas de texto
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.3), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 3. Detectar bordas nas linhas horizontais
    edges = cv2.Canny(horizontal_lines, 50, 150, apertureSize=3)
    
    # 4. Usar HoughLinesP (probabilístico) - mais eficiente e preciso
    # Parâmetros ajustados para detectar linhas horizontais longas
    min_line_length = int(w * 0.2)  # Linha deve ter pelo menos 20% da largura
    max_line_gap = int(h * 0.02)     # Gap máximo entre segmentos de linha
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(w * 0.15),  # Threshold adaptativo baseado na largura
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None or len(lines) == 0:
        # Fallback: tentar método alternativo com Canny direto
        edges_fallback = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges_fallback,
            rho=1,
            theta=np.pi / 180,
            threshold=int(w * 0.1),
            minLineLength=int(w * 0.1),
            maxLineGap=int(h * 0.05)
        )
        
        if lines is None or len(lines) == 0:
            return image, 0.0

    # 5. Calcular ângulos das linhas e filtrar horizontais
    angles = []
    weights = []  # Peso baseado no comprimento da linha
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcular ângulo da linha
        if abs(x2 - x1) < 1:  # Linha quase vertical, ignorar
            continue
            
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        
        # Focar em linhas quase horizontais (entre -30 e 30 graus)
        if abs(angle_deg) > 30:
            continue
        
        # Comprimento da linha (peso)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        angles.append(angle_deg)
        weights.append(length)

    if not angles:
        return image, 0.0

    # 6. Calcular ângulo final usando média ponderada
    # Linhas mais longas têm mais peso (são mais confiáveis)
    angles_array = np.array(angles)
    weights_array = np.array(weights)
    
    # Normalizar pesos
    if weights_array.sum() > 0:
        weights_array = weights_array / weights_array.sum()
        weighted_angle = np.average(angles_array, weights=weights_array)
    else:
        weighted_angle = np.median(angles_array)
    
    # 7. Filtrar outliers usando IQR (Interquartile Range)
    q1 = np.percentile(angles_array, 25)
    q3 = np.percentile(angles_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered_angles = [a for a, w in zip(angles, weights) 
                      if lower_bound <= a <= upper_bound]
    filtered_weights = [w for a, w in zip(angles, weights) 
                       if lower_bound <= a <= upper_bound]
    
    if filtered_angles:
        filtered_angles_array = np.array(filtered_angles)
        filtered_weights_array = np.array(filtered_weights)
        
        if filtered_weights_array.sum() > 0:
            filtered_weights_array = filtered_weights_array / filtered_weights_array.sum()
            final_angle = np.average(filtered_angles_array, weights=filtered_weights_array)
        else:
            final_angle = np.median(filtered_angles_array)
    else:
        final_angle = weighted_angle

    # 8. Só corrige se exceder threshold
    if abs(final_angle) < threshold:
        return image, final_angle

    # 9. Aplica rotação com interpolação de alta qualidade
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, final_angle, 1.0)
    
    # Calcular novo tamanho para evitar cortes
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Ajustar matriz de rotação para o novo centro
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
    )

    return rotated, final_angle


def binarize_image(image: np.ndarray, method: str = "adaptive", block_size: int = 11) -> np.ndarray:
    """
    Binariza imagem.

    Args:
        image: Imagem em escala de cinza
        method: Método ('otsu', 'adaptive', 'none')
        block_size: Tamanho do bloco para método adaptativo

    Returns:
        Imagem binarizada
    """
    validate_image(image)
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image

    if method == "none":
        return gray

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    if method == "adaptive":
        # Garante que block_size é ímpar
        if block_size % 2 == 0:
            block_size += 1
        # C reduzido de 2 para 5 para preservar mais detalhes e reduzir granulação
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 5
        )
        return binary

    raise ValueError(f"Método de binarização inválido: {method}")


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Reduz ruído da imagem preservando detalhes importantes.

    Args:
        image: Imagem
        strength: Força da redução (1-20), valores menores preservam mais detalhes

    Returns:
        Imagem com ruído reduzido
    """
    validate_image(image)

    if len(image.shape) == 3:
        # Colorida: usa filtro bilateral com parâmetros mais suaves
        # Reduz o d (diâmetro) e os valores de sigma para preservar mais detalhes
        d = max(5, min(9, 5 + strength // 2))  # Diâmetro entre 5-9 baseado na força
        sigma_color = max(20, strength * 8)  # Reduzido de strength * 10
        sigma_space = max(20, strength * 8)  # Reduzido de strength * 10
        denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    else:
        # Escala de cinza: usa filtro não-local means com parâmetros mais suaves
        # h reduzido para preservar mais detalhes quando strength é baixo
        h_value = max(3.0, min(10.0, strength * 0.8))  # Mais suave que strength direto
        denoised = cv2.fastNlMeansDenoising(
            image, None, h=h_value, templateWindowSize=7, searchWindowSize=21
        )

    return denoised


def sharpen_image(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """
    Aplica sharpening (unsharp mask) para recuperar nitidez do texto impresso.

    O sharpening é aplicado após o denoise para recuperar detalhes finos
    do texto impresso pequeno, mantendo a suavidade do manuscrito.

    Args:
        image: Imagem
        strength: Força do sharpening (0.1-2.0), valores menores são mais suaves

    Returns:
        Imagem com nitidez recuperada
    """
    validate_image(image)

    # Converte para float32 para operações matemáticas
    if image.dtype != np.float32:
        img_float = image.astype(np.float32)
    else:
        img_float = image.copy()

    # Aplica blur gaussiano (unsharp mask)
    # Usa kernel pequeno (5x5) para preservar detalhes finos
    # O kernel é calculado automaticamente baseado no sigma
    blurred = cv2.GaussianBlur(img_float, (5, 5), sigmaX=1.0, sigmaY=1.0)

    # Calcula a máscara (diferença entre original e borrado)
    mask = img_float - blurred

    # Aplica o sharpening: original + (mask * strength)
    # Strength controla quanto da diferença é adicionada de volta
    sharpened = img_float + (mask * strength)

    # Garante que os valores ficam no range válido
    if len(image.shape) == 3:
        sharpened = np.clip(sharpened, 0, 255)
    else:
        sharpened = np.clip(sharpened, 0, 255)

    # Converte de volta para uint8
    return sharpened.astype(np.uint8)


def normalize_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Normaliza contraste usando CLAHE preservando detalhes.

    Args:
        image: Imagem
        clip_limit: Limite de clip para CLAHE (valores menores preservam mais detalhes)

    Returns:
        Imagem com contraste normalizado
    """
    validate_image(image)

    # Tile grid size maior reduz artefatos e granulação
    tile_size = (8, 8)

    if len(image.shape) == 3:
        # Colorida: aplica CLAHE em cada canal
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Escala de cinza
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return clahe.apply(image)


def remove_borders(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Remove bordas pretas da imagem.

    Args:
        image: Imagem
        threshold: Threshold de pixels pretos

    Returns:
        Imagem sem bordas
    """
    validate_image(image)
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image

    # Detecta bordas pretas
    mask = gray > threshold
    coords = np.column_stack(np.where(mask))

    if len(coords) == 0:
        return image

    # Encontra bounding box do conteúdo
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop da região válida
    if len(image.shape) == 3:
        return image[y_min:y_max, x_min:x_max]
    return image[y_min:y_max, x_min:x_max]


def preprocess_pipeline(
    image: np.ndarray, config: PreprocessingConfig, debug_dir: Path | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Aplica pipeline completo de pré-processamento.

    Args:
        image: Imagem original
        config: Configuração de pré-processamento
        debug_dir: Diretório para salvar imagens intermediárias (opcional)

    Returns:
        Tupla (imagem processada, metadados)
    """
    validate_image(image)
    metadata: dict[str, Any] = {}
    processed = image.copy()

    # Deskew
    if config.deskew:
        processed, angle = deskew_image(processed, config.deskew_threshold)
        metadata["deskew_angle"] = angle
        if debug_dir:
            cv2.imwrite(str(debug_dir / "01_deskewed.jpg"), processed)

    # Binarização
    if config.binarization != "none":
        processed = binarize_image(processed, config.binarization, config.adaptive_block_size)
        if debug_dir:
            cv2.imwrite(str(debug_dir / "02_binarized.jpg"), processed)

    # Denoise
    if config.denoise:
        processed = denoise_image(processed, config.denoise_strength)
        if debug_dir:
            cv2.imwrite(str(debug_dir / "03_denoised.jpg"), processed)

    # Sharpening (após denoise para recuperar nitidez do texto impresso)
    if config.sharpen:
        processed = sharpen_image(processed, config.sharpen_strength)
        if debug_dir:
            cv2.imwrite(str(debug_dir / "03b_sharpened.jpg"), processed)

    # Normalização de contraste
    if config.contrast_normalization:
        processed = normalize_contrast(processed, config.clahe_clip_limit)
        if debug_dir:
            cv2.imwrite(str(debug_dir / "04_contrast_normalized.jpg"), processed)

    # Remoção de bordas
    if config.border_removal:
        processed = remove_borders(processed, config.border_threshold)
        if debug_dir:
            cv2.imwrite(str(debug_dir / "05_borders_removed.jpg"), processed)

    metadata["final_shape"] = processed.shape
    return processed, metadata


