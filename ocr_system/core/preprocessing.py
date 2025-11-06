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


def _deskew_hough_lines(image: np.ndarray) -> float | None:
    """
    Detecta ângulo de inclinação usando HoughLinesP.
    
    Args:
        image: Imagem em escala de cinza
        
    Returns:
        Ângulo detectado em graus ou None se não detectado
    """
    h, w = image.shape
    
    # Binarizar para focar em linhas pretas (texto/documento)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morfologia para destacar linhas horizontais
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.3), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detectar bordas
    edges = cv2.Canny(horizontal_lines, 50, 150, apertureSize=3)
    
    # HoughLinesP - ajustado para detectar linhas mais longas e melhorar precisão
    min_line_length = int(w * 0.3)  # Aumentado para detectar linhas mais longas
    max_line_gap = int(h * 0.01)  # Reduzido para linhas mais contínuas
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,  # 1 grau de precisão
        threshold=int(w * 0.1),  # Reduzido para detectar mais linhas
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None or len(lines) == 0:
        # Fallback: tentar com Canny direto
        edges_fallback = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges_fallback,
            rho=1,
            theta=np.pi / 180,
            threshold=int(w * 0.1),
            minLineLength=int(w * 0.1),
            maxLineGap=int(h * 0.05)
        )
        
        if lines is None or len(lines) == 0:
            return None

    # Calcular ângulos das linhas
    angles = []
    weights = []
    
    # Zona de exclusão para linhas de erro de escaneamento (últimos 15% da largura)
    exclude_zone_start = int(w * 0.85)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        if abs(x2 - x1) < 1:
            continue
        
        # FILTRO: Ignora linhas que estão na zona de erro de escaneamento
        # Se a linha está na zona de exclusão, ignora completamente
        line_center_x = (x1 + x2) / 2.0
        if line_center_x > exclude_zone_start:
            # Linha na zona de erro - ignora completamente
            continue
        
        # FILTRO: Prefere linhas no centro da imagem (mais confiáveis)
        center_weight = 1.0
        distance_from_center = abs(line_center_x - (w / 2.0))
        # Linhas mais próximas do centro têm peso maior
        center_weight = 1.0 - (distance_from_center / (w / 2.0)) * 0.5
        center_weight = max(0.5, center_weight)  # Mínimo 50% do peso
            
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        
        # Filtra apenas ângulos próximos de 0 (linhas horizontais)
        # Aceita até 7 graus para capturar inclinações pequenas e médias
        if abs(angle_deg) > 7.0:
            continue
        
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Aplica peso do centro ao comprimento
        weighted_length = length * center_weight
        angles.append(angle_deg)
        weights.append(weighted_length)

    if not angles:
        return None

    # Filtrar outliers usando IQR (mais agressivo para remover linhas defeituosas)
    angles_array = np.array(angles)
    weights_array = np.array(weights)
    
    q1 = np.percentile(angles_array, 25)
    q3 = np.percentile(angles_array, 75)
    iqr = q3 - q1
    # FILTRO MELHORADO: Usa 1.0 * IQR em vez de 1.5 para ser mais restritivo
    lower_bound = q1 - 1.0 * iqr
    upper_bound = q3 + 1.0 * iqr
    
    filtered_angles = [a for a, w in zip(angles, weights) 
                      if lower_bound <= a <= upper_bound]
    filtered_weights = [w for a, w in zip(angles, weights) 
                       if lower_bound <= a <= upper_bound]
    
    if filtered_angles:
        filtered_angles_array = np.array(filtered_angles)
        filtered_weights_array = np.array(filtered_weights)
        
        if filtered_weights_array.sum() > 0:
            filtered_weights_array = filtered_weights_array / filtered_weights_array.sum()
            return float(np.average(filtered_angles_array, weights=filtered_weights_array))
        else:
            return float(np.median(filtered_angles_array))
    
    if weights_array.sum() > 0:
        weights_array = weights_array / weights_array.sum()
        return float(np.average(angles_array, weights=weights_array))
    
    return float(np.median(angles_array))


def _deskew_projection_profile(image: np.ndarray, angle_range: tuple[float, float] = (-1.0, 1.0)) -> float | None:
    """
    Detecta ângulo de inclinação usando Projection Profile.
    
    O método testa diferentes ângulos e escolhe o que maximiza a variância
    do perfil de projeção horizontal (mais linhas de texto bem definidas).
    
    Args:
        image: Imagem em escala de cinza
        angle_range: Range de ângulos para testar (min, max) em graus
        
    Returns:
        Ângulo detectado em graus ou None se não detectado
    """
    h, w = image.shape
    
    # Binarizar
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # FILTRO: Remove zona de erro de escaneamento antes de testar
    exclude_zone_start = int(w * 0.85)
    # Cria máscara para ignorar a zona de erro
    mask = np.ones(binary.shape, dtype=np.uint8) * 255
    mask[:, exclude_zone_start:] = 0  # Zera a zona de erro
    binary = cv2.bitwise_and(binary, mask)
    
    # Testar diferentes ângulos com maior precisão (0.05 graus para detectar 0.56°)
    angles_to_test = np.arange(angle_range[0], angle_range[1] + 0.05, 0.05)
    best_angle = 0.0
    best_variance = 0.0
    
    center = (w // 2, h // 2)
    
    for angle in angles_to_test:
        # Rotacionar imagem
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        rotated = cv2.warpAffine(
            binary, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        # Calcular perfil de projeção horizontal (soma de pixels por linha)
        projection = np.sum(rotated, axis=1)
        
        # Calcular variância (maior variância = linhas de texto mais definidas)
        variance = float(np.var(projection))
        
        if variance > best_variance:
            best_variance = variance
            best_angle = angle
    
    # Só retorna se a variância for significativa
    if best_variance > 1000:  # Threshold mínimo
        return best_angle
    
    return None


def _deskew_connected_components(image: np.ndarray) -> float | None:
    """
    Detecta ângulo de inclinação usando análise de componentes conectados.
    
    Analisa a orientação dos componentes de texto para determinar a inclinação.
    
    Args:
        image: Imagem em escala de cinza
        
    Returns:
        Ângulo detectado em graus ou None se não detectado
    """
    h, w = image.shape
    
    # Binarizar
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Encontrar componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    if num_labels < 2:
        return None
    
    # Filtrar componentes pequenos (ruído)
    min_area = (h * w) * 0.0001  # 0.01% da imagem
    angles = []
    weights = []
    
    for i in range(1, num_labels):  # Pular background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        
        # Extrair componente
        component_mask = (labels == i).astype(np.uint8) * 255
        
        # Encontrar contornos
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Usar maior contorno
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Ajustar retângulo rotacionado
        if len(largest_contour) < 5:
            continue
        
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Normalizar ângulo para -45 a 45 graus
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        # Filtrar ângulos muito inclinados
        if abs(angle) > 30:
            continue
        
        angles.append(angle)
        weights.append(area)
    
    if not angles:
        return None
    
    # Média ponderada
    angles_array = np.array(angles)
    weights_array = np.array(weights)
    
    if weights_array.sum() > 0:
        weights_array = weights_array / weights_array.sum()
        return float(np.average(angles_array, weights=weights_array))
    
    return float(np.median(angles_array))


def deskew_image(image: np.ndarray, threshold: float = 0.1) -> tuple[np.ndarray, float]:
    """
    Corrige rotação da imagem (deskew) usando múltiplas estratégias.

    Usa três métodos diferentes e combina os resultados:
    1. HoughLinesP: Detecta linhas horizontais
    2. Projection Profile: Maximiza variância do perfil de projeção
    3. Componentes Conectados: Analisa orientação de componentes de texto

    Args:
        image: Imagem em escala de cinza ou colorida
        threshold: Graus mínimos para aplicar correção (padrão: 0.1)

    Returns:
        Tupla (imagem corrigida, ângulo detectado)
    """
    validate_image(image)
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Tentar múltiplas estratégias
    angles = []
    weights = []
    
    # 1. HoughLinesP (peso alto - mais confiável)
    hough_angle = _deskew_hough_lines(gray)
    if hough_angle is not None:
        angles.append(hough_angle)
        weights.append(3.0)  # Peso maior
    
    # 2. Projection Profile (peso médio)
    proj_angle = _deskew_projection_profile(gray)
    if proj_angle is not None:
        angles.append(proj_angle)
        weights.append(2.0)
    
    # 3. Componentes Conectados (peso baixo - pode ser ruidoso)
    cc_angle = _deskew_connected_components(gray)
    if cc_angle is not None:
        angles.append(cc_angle)
        weights.append(1.0)
    
    if not angles:
        return image, 0.0
    
    # Calcular ângulo final usando média ponderada
    angles_array = np.array(angles)
    weights_array = np.array(weights)
    
    if weights_array.sum() > 0:
        weights_array = weights_array / weights_array.sum()
        final_angle = float(np.average(angles_array, weights=weights_array))
    else:
        final_angle = float(np.median(angles_array))
    
    # Validar ângulo (não deve ser muito extremo)
    if abs(final_angle) > 7.0:  # Limite máximo de 7 graus
        # Se o ângulo for muito grande, provavelmente é um erro de detecção
        return image, final_angle
    
    # Só corrige se exceder threshold (reduzido para detectar 0.56°)
    # Threshold mínimo de 0.3 graus para capturar inclinações pequenas
    min_threshold = min(threshold, 0.3)
    if abs(final_angle) < min_threshold:
        return image, final_angle
    
    # Permite correção até 7 graus para corrigir inclinações maiores
    if abs(final_angle) > 7.0:  # Limita a máximo 7 graus
        final_angle = np.sign(final_angle) * min(7.0, abs(final_angle))

    # Aplica rotação com interpolação de alta qualidade
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


