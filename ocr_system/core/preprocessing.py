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


def _detect_border_ghost_lines(
    image: np.ndarray, border_percent: float = 0.1
) -> np.ndarray:
    """
    Detecta linhas fantasma horizontais e verticais nas bordas da imagem.
    
    Detecta linhas que aparecem nos primeiros/últimos N% da largura/altura,
    que são geralmente artefatos de escaneamento.
    
    Args:
        image: Imagem em escala de cinza
        border_percent: Percentual das bordas a considerar (0.05 = 5%, 0.1 = 10%)
        
    Returns:
        Máscara binária (uint8) onde pixels de linhas fantasma são 255
    """
    validate_image(image)
    h, w = image.shape
    
    # Calcular zonas de borda
    border_w = int(w * border_percent)
    border_h = int(h * border_percent)
    
    # Binarizar para focar em linhas escuras
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Criar máscara inicial (zeros = não processar, 255 = processar)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Marcar zonas de borda na máscara
    # Borda esquerda
    mask[:, :border_w] = 255
    # Borda direita
    mask[:, w - border_w:] = 255
    # Borda superior
    mask[:border_h, :] = 255
    # Borda inferior
    mask[h - border_h:, :] = 255
    
    # Aplicar máscara na imagem binarizada
    border_region = cv2.bitwise_and(binary, mask)
    
    # Detectar linhas horizontais nas bordas
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.2), 1))
    horizontal_lines = cv2.morphologyEx(border_region, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detectar linhas verticais nas bordas
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.2)))
    vertical_lines = cv2.morphologyEx(border_region, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combinar linhas horizontais e verticais
    ghost_lines_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Usar HoughLinesP para refinar detecção nas bordas
    edges = cv2.Canny(ghost_lines_mask, 50, 150, apertureSize=3)
    
    # Detectar linhas horizontais
    min_line_length_h = int(w * 0.15)  # Linhas devem ter pelo menos 15% da largura
    lines_h = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(w * 0.05),
        minLineLength=min_line_length_h,
        maxLineGap=int(h * 0.02)
    )
    
    # Detectar linhas verticais
    min_line_length_v = int(h * 0.15)  # Linhas devem ter pelo menos 15% da altura
    lines_v = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(h * 0.05),
        minLineLength=min_line_length_v,
        maxLineGap=int(w * 0.02)
    )
    
    # Criar máscara final de linhas fantasma
    result_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Desenhar linhas horizontais detectadas
    if lines_h is not None:
        for line in lines_h:
            x1, y1, x2, y2 = line[0]
            # Verificar se a linha está realmente na borda
            line_center_y = (y1 + y2) / 2.0
            if line_center_y < border_h or line_center_y > (h - border_h):
                cv2.line(result_mask, (x1, y1), (x2, y2), 255, 3)
    
    # Desenhar linhas verticais detectadas
    if lines_v is not None:
        for line in lines_v:
            x1, y1, x2, y2 = line[0]
            # Verificar se a linha está realmente na borda
            line_center_x = (x1 + x2) / 2.0
            if line_center_x < border_w or line_center_x > (w - border_w):
                cv2.line(result_mask, (x1, y1), (x2, y2), 255, 3)
    
    # Dilatar máscara para cobrir toda a espessura da linha
    kernel = np.ones((5, 5), np.uint8)
    result_mask = cv2.dilate(result_mask, kernel, iterations=2)
    
    return result_mask


def _remove_border_ghost_lines(
    image: np.ndarray, border_percent: float = 0.1
) -> np.ndarray:
    """
    Remove linhas fantasma das bordas da imagem usando inpaint.
    
    Args:
        image: Imagem (pode ser colorida ou escala de cinza)
        border_percent: Percentual das bordas a considerar (0.05 = 5%, 0.1 = 10%)
        
    Returns:
        Imagem com linhas fantasma removidas
    """
    validate_image(image)
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image
    
    # Detectar linhas fantasma
    ghost_mask = _detect_border_ghost_lines(gray, border_percent)
    
    # Se não detectou linhas, retorna imagem original
    if np.sum(ghost_mask) == 0:
        return image
    
    # Aplicar inpaint para remover linhas
    if len(image.shape) == 3:
        # Imagem colorida: aplicar inpaint em cada canal
        result = image.copy()
        for channel in range(3):
            result[:, :, channel] = cv2.inpaint(
                image[:, :, channel],
                ghost_mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
        return result
    else:
        # Imagem em escala de cinza
        return cv2.inpaint(
            image,
            ghost_mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )


def _detect_long_crossing_lines(
    image: np.ndarray, min_length_percent: float = 0.8
) -> np.ndarray:
    """
    Detecta linhas longas que atravessam a imagem (horizontais ou verticais).
    
    Detecta linhas que têm comprimento > min_length_percent da dimensão da imagem.
    Filtra linhas que não são parte de tabelas baseado em características.
    
    Args:
        image: Imagem em escala de cinza
        min_length_percent: Percentual mínimo de comprimento (0.8 = 80%)
        
    Returns:
        Máscara binária (uint8) onde pixels de linhas longas são 255
    """
    validate_image(image)
    h, w = image.shape
    
    # Binarizar
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detectar linhas horizontais
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.5), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detectar linhas verticais
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.5)))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combinar
    all_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Detectar bordas
    edges = cv2.Canny(all_lines, 50, 150, apertureSize=3)
    
    # Detectar linhas com HoughLinesP
    min_line_length_h = int(w * min_length_percent)
    min_line_length_v = int(h * min_length_percent)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(min(w, h) * 0.1),
        minLineLength=min(min_line_length_h, min_line_length_v),
        maxLineGap=int(min(w, h) * 0.02)
    )
    
    if lines is None:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Criar máscara de linhas longas
    result_mask = np.zeros((h, w), dtype=np.uint8)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcular comprimento e ângulo
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        
        # Verificar se é linha horizontal (ângulo próximo de 0 ou 180)
        is_horizontal = abs(angle_deg) < 10 or abs(angle_deg) > 170
        # Verificar se é linha vertical (ângulo próximo de 90 ou -90)
        is_vertical = 80 < abs(angle_deg) < 100
        
        if is_horizontal:
            # Linha horizontal: verificar se tem comprimento suficiente
            if length >= min_line_length_h:
                # Verificar se não é linha de tabela válida
                # Linhas de tabela geralmente têm espessura consistente e posição regular
                # Linhas fantasma são mais finas e irregulares
                line_thickness = _estimate_line_thickness(binary, x1, y1, x2, y2, horizontal=True)
                
                # Se a linha é muito fina ou muito espessa, pode ser fantasma
                # Linhas de tabela têm espessura entre 2-5 pixels tipicamente
                if line_thickness < 1.5 or line_thickness > 8:
                    cv2.line(result_mask, (x1, y1), (x2, y2), 255, 3)
        
        elif is_vertical:
            # Linha vertical: verificar se tem comprimento suficiente
            if length >= min_line_length_v:
                # Verificar espessura
                line_thickness = _estimate_line_thickness(binary, x1, y1, x2, y2, horizontal=False)
                
                if line_thickness < 1.5 or line_thickness > 8:
                    cv2.line(result_mask, (x1, y1), (x2, y2), 255, 3)
    
    # Dilatar máscara
    kernel = np.ones((5, 5), np.uint8)
    result_mask = cv2.dilate(result_mask, kernel, iterations=2)
    
    return result_mask


def _estimate_line_thickness(
    binary: np.ndarray, x1: int, y1: int, x2: int, y2: int, horizontal: bool
) -> float:
    """
    Estima a espessura de uma linha analisando pixels perpendiculares.
    
    Args:
        binary: Imagem binarizada
        x1, y1, x2, y2: Coordenadas da linha
        horizontal: True se linha horizontal, False se vertical
        
    Returns:
        Espessura média estimada em pixels
    """
    h, w = binary.shape
    thicknesses = []
    
    # Amostrar alguns pontos ao longo da linha
    num_samples = 10
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        
        if horizontal:
            # Para linha horizontal, verificar espessura vertical
            # Contar pixels pretos acima e abaixo
            thickness = 0
            for dy in range(-5, 6):
                ny = y + dy
                if 0 <= ny < h and binary[ny, x] == 0:
                    thickness += 1
        else:
            # Para linha vertical, verificar espessura horizontal
            thickness = 0
            for dx in range(-5, 6):
                nx = x + dx
                if 0 <= nx < w and binary[y, nx] == 0:
                    thickness += 1
        
        if thickness > 0:
            thicknesses.append(thickness)
    
    return float(np.mean(thicknesses)) if thicknesses else 0.0


def _remove_long_crossing_lines(
    image: np.ndarray, min_length_percent: float = 0.8
) -> np.ndarray:
    """
    Remove linhas longas que atravessam a imagem usando inpaint.
    
    Args:
        image: Imagem (pode ser colorida ou escala de cinza)
        min_length_percent: Percentual mínimo de comprimento (0.8 = 80%)
        
    Returns:
        Imagem com linhas longas removidas
    """
    validate_image(image)
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image
    
    # Detectar linhas longas
    lines_mask = _detect_long_crossing_lines(gray, min_length_percent)
    
    # Se não detectou linhas, retorna imagem original
    if np.sum(lines_mask) == 0:
        return image
    
    # Aplicar inpaint
    if len(image.shape) == 3:
        result = image.copy()
        for channel in range(3):
            result[:, :, channel] = cv2.inpaint(
                image[:, :, channel],
                lines_mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
        return result
    else:
        return cv2.inpaint(
            image,
            lines_mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )


def _identify_text_regions(image: np.ndarray) -> np.ndarray:
    """
    Identifica regiões de texto usando análise morfológica.
    
    Filtra componentes por área e aspect ratio para distinguir texto de linhas
    de tabela e outros artefatos.
    
    Args:
        image: Imagem em escala de cinza
        
    Returns:
        Máscara binária (uint8) onde regiões de texto são 255
    """
    validate_image(image)
    h, w = image.shape
    
    # Binarizar
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Operações morfológicas para conectar caracteres em palavras
    # Kernel horizontal para conectar caracteres na mesma linha
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.05), 1))
    text_regions = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)
    
    # Encontrar componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        text_regions, connectivity=8
    )
    
    if num_labels < 2:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Calcular limites de área para texto
    # Texto típico: área entre 0.01% e 5% da imagem
    min_area = (h * w) * 0.0001  # 0.01% da imagem
    max_area = (h * w) * 0.05  # 5% da imagem
    
    # Criar máscara de texto
    text_mask = np.zeros((h, w), dtype=np.uint8)
    
    for i in range(1, num_labels):  # Pular background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtrar por área
        if area < min_area or area > max_area:
            continue
        
        # Obter dimensões do componente
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Calcular aspect ratio
        if height > 0:
            aspect_ratio = width / height
        else:
            continue
        
        # Texto geralmente tem aspect ratio > 1 (mais largo que alto)
        # Linhas de tabela têm aspect ratio muito alto (>10) ou muito baixo (<0.1)
        # Filtrar linhas muito longas e finas (tabelas)
        if aspect_ratio > 20 or aspect_ratio < 0.05:
            continue
        
        # Filtrar componentes muito pequenos em uma dimensão (ruído)
        if width < 5 or height < 5:
            continue
        
        # Filtrar componentes muito grandes (provavelmente não é texto)
        if width > w * 0.8 or height > h * 0.3:
            continue
        
        # Adicionar componente à máscara de texto
        component_mask = (labels == i).astype(np.uint8) * 255
        text_mask = cv2.bitwise_or(text_mask, component_mask)
    
    # Aplicar morfologia para suavizar a máscara
    kernel = np.ones((3, 3), np.uint8)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return text_mask


def _filter_table_lines(image: np.ndarray) -> np.ndarray:
    """
    Detecta e filtra linhas de tabela para exclusão do deskew.
    
    Classifica linhas como "tabela" baseado em comprimento, espessura e posição.
    
    Args:
        image: Imagem em escala de cinza
        
    Returns:
        Máscara binária (uint8) onde linhas de tabela são 255
    """
    validate_image(image)
    h, w = image.shape
    
    # Binarizar
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Detectar linhas horizontais
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.3), 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Detectar linhas verticais
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h * 0.3)))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combinar
    all_lines = cv2.bitwise_or(horizontal_lines, vertical_lines)
    
    # Detectar bordas
    edges = cv2.Canny(all_lines, 50, 150, apertureSize=3)
    
    # Detectar linhas com HoughLinesP
    min_line_length_h = int(w * 0.5)  # Linhas de tabela são longas
    min_line_length_v = int(h * 0.5)
    
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(min(w, h) * 0.1),
        minLineLength=min(min_line_length_h, min_line_length_v),
        maxLineGap=int(min(w, h) * 0.02)
    )
    
    if lines is None:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Criar máscara de linhas de tabela
    table_mask = np.zeros((h, w), dtype=np.uint8)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calcular comprimento e ângulo
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad)
        
        # Verificar se é linha horizontal
        is_horizontal = abs(angle_deg) < 10 or abs(angle_deg) > 170
        # Verificar se é linha vertical
        is_vertical = 80 < abs(angle_deg) < 100
        
        if is_horizontal:
            # Linha horizontal: verificar se tem comprimento suficiente
            if length >= min_line_length_h:
                # Verificar espessura (linhas de tabela têm espessura consistente)
                line_thickness = _estimate_line_thickness(binary, x1, y1, x2, y2, horizontal=True)
                
                # Linhas de tabela têm espessura entre 2-5 pixels tipicamente
                if 1.5 <= line_thickness <= 8:
                    cv2.line(table_mask, (x1, y1), (x2, y2), 255, 3)
        
        elif is_vertical:
            # Linha vertical: verificar se tem comprimento suficiente
            if length >= min_line_length_v:
                # Verificar espessura
                line_thickness = _estimate_line_thickness(binary, x1, y1, x2, y2, horizontal=False)
                
                if 1.5 <= line_thickness <= 8:
                    cv2.line(table_mask, (x1, y1), (x2, y2), 255, 3)
    
    # Dilatar máscara
    kernel = np.ones((5, 5), np.uint8)
    table_mask = cv2.dilate(table_mask, kernel, iterations=2)
    
    return table_mask


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


def _deskew_text_dominant(
    image: np.ndarray, angle_range: tuple[float, float] = (-1.0, 1.0)
) -> float | None:
    """
    Detecta ângulo de inclinação baseado apenas em regiões de texto dominante.
    
    Usa análise morfológica para identificar texto, exclui linhas de tabela e bordas,
    e aplica Projection Profile apenas nas regiões de texto.
    
    Args:
        image: Imagem em escala de cinza
        angle_range: Range de ângulos para testar (min, max) em graus
        
    Returns:
        Ângulo detectado em graus ou None se não detectado
    """
    validate_image(image)
    h, w = image.shape
    
    # Identificar regiões de texto
    text_mask = _identify_text_regions(image)
    
    # Se não encontrou texto suficiente, retorna None
    text_pixels = np.sum(text_mask > 0)
    if text_pixels < (h * w) * 0.01:  # Menos de 1% da imagem é texto
        return None
    
    # Filtrar linhas de tabela da máscara de texto
    table_mask = _filter_table_lines(image)
    # Remover linhas de tabela da máscara de texto
    text_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(table_mask))
    
    # Binarizar imagem
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Aplicar máscara de texto na imagem binarizada
    # Só processar regiões de texto
    text_only = cv2.bitwise_and(binary, text_mask)
    
    # Testar diferentes ângulos com precisão de 0.05 graus
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
        
        # Rotacionar imagem binarizada com texto
        rotated = cv2.warpAffine(
            text_only, rotation_matrix, (new_w, new_h),
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


def deskew_image(
    image: np.ndarray, threshold: float = 0.1, use_text_dominant: bool = False
) -> tuple[np.ndarray, float]:
    """
    Corrige rotação da imagem (deskew) usando múltiplas estratégias.

    Se use_text_dominant=True, usa apenas análise morfológica de texto dominante.
    Caso contrário, usa três métodos diferentes e combina os resultados:
    1. HoughLinesP: Detecta linhas horizontais
    2. Projection Profile: Maximiza variância do perfil de projeção
    3. Componentes Conectados: Analisa orientação de componentes de texto

    Args:
        image: Imagem em escala de cinza ou colorida
        threshold: Graus mínimos para aplicar correção (padrão: 0.1)
        use_text_dominant: Se True, usa apenas deskew baseado em texto dominante

    Returns:
        Tupla (imagem corrigida, ângulo detectado)
    """
    validate_image(image)
    gray = image_to_grayscale(image) if len(image.shape) == 3 else image
    h, w = gray.shape

    # Se usar texto dominante, tentar apenas esse método primeiro
    if use_text_dominant:
        text_angle = _deskew_text_dominant(gray)
        if text_angle is not None:
            # Usar apenas o ângulo do texto dominante
            angles = [text_angle]
            weights = [1.0]
        else:
            # Fallback para métodos tradicionais
            angles = []
            weights = []
    else:
        angles = []
        weights = []
    
    # Métodos tradicionais (usados se não usar texto dominante ou como fallback)
    if not use_text_dominant or not angles:
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

    # Remoção de linhas fantasma/bordas de scanner (ANTES do deskew)
    if config.remove_ghost_lines:
        processed = _remove_border_ghost_lines(processed, config.ghost_line_border_percent)
        processed = _remove_long_crossing_lines(processed, min_length_percent=0.8)
        if debug_dir:
            cv2.imwrite(str(debug_dir / "00_ghost_lines_removed.jpg"), processed)

    # Deskew
    if config.deskew:
        processed, angle = deskew_image(
            processed,
            config.deskew_threshold,
            use_text_dominant=config.deskew_text_dominant
        )
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


