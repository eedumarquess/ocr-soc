"""Ferramenta interativa para criar e ajustar ROIs em layouts."""

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ocr_system.core.anchor_detector import detect_anchor_with_fallback
from ocr_system.core.ocr_engine import initialize_ocr
from ocr_system.core.preprocessing import preprocess_pipeline
from ocr_system.core.processor import load_layout
from ocr_system.models.anchor import QRCodeAnchorConfig, TextAnchorConfig
from ocr_system.models.layout import LayoutConfig, PreprocessingConfig
from ocr_system.models.roi import OCRConfig, ROIConfig, ValidationConfig
from ocr_system.utils.exceptions import AnchorNotFoundError
from ocr_system.utils.image_io import load_image


class ROISelector:
    """Editor interativo de ROIs com coordenadas relativas à âncora."""

    def __init__(
        self,
        image: np.ndarray,
        layout: LayoutConfig,
        layout_path: Path,
        processed_image: np.ndarray | None = None,
        anchor_position: tuple[int, int] | None = None,
    ):
        self.original_image = image.copy()
        self.processed_image = processed_image if processed_image is not None else image.copy()
        self.layout = layout
        self.layout_path = layout_path
        self.anchor_position = anchor_position

        # Estado dos ROIs em coordenadas absolutas (para visualização)
        self.roi_boxes: list[dict[str, Any]] = []
        self.current_roi_idx = -1
        self.dragging = False
        self.drag_handle: Optional[str] = None
        self.window_name = "ROI Editor - Pressione 'h' para ajuda"

        # Modo de criação de novo ROI
        self.creating_new_roi = False
        self.new_roi_start: Optional[tuple[int, int]] = None
        self.new_roi_box: Optional[dict[str, Any]] = None

        # Modo de definição manual de âncora
        self.waiting_for_anchor_click = False

        # Zoom e pan
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.viewport_width = 1400
        self.viewport_height = 900

        # Atualizar boxes dos ROIs
        self.update_roi_boxes()

    def update_roi_boxes(self) -> None:
        """Atualiza as caixas dos ROIs em coordenadas absolutas."""
        if not self.anchor_position:
            self.roi_boxes = []
            return

        h, w = self.processed_image.shape[:2]
        anchor_x, anchor_y = self.anchor_position
        self.roi_boxes = []

        for roi in self.layout.rois:
            offset_x, offset_y = roi.relative_position
            width, height = roi.size

            # Calcular posição absoluta
            abs_x = anchor_x + offset_x
            abs_y = anchor_y + offset_y

            # Garantir limites
            abs_x = max(0, min(abs_x, w - 1))
            abs_y = max(0, min(abs_y, h - 1))
            width = min(width, w - abs_x)
            height = min(height, h - abs_y)

            self.roi_boxes.append(
                {
                    "x": abs_x,
                    "y": abs_y,
                    "w": width,
                    "h": height,
                    "roi": roi,
                }
            )

    def detect_anchor_interactive(self) -> bool:
        """Detecta âncora interativamente."""
        if not self.layout.anchor:
            print("[ERRO] Layout não possui configuração de âncora!")
            return False

        print("\n[INFO] Detectando âncora...")
        print(f"     Estratégia primária: {self.layout.anchor.primary.type}")
        print(f"     Fallbacks disponíveis: {len(self.layout.anchor.fallbacks)}")
        print(f"     Fallback para imagem original: {self.layout.anchor.try_original_on_failure}")
        
        # Usa função compartilhada com fallback configurável
        try:
            ocr_instance = initialize_ocr()
            print("     Tentando detectar...")
            position, confidence, a_type, metadata = detect_anchor_with_fallback(
                self.processed_image, self.original_image, self.layout.anchor, ocr_instance
            )
            self.anchor_position = position
            print(f"[OK] Âncora detectada: tipo={a_type}, posição={position}, confiança={confidence:.2f}")
            
            # Mostra em qual imagem foi detectada
            if metadata:
                if "detected_on_processed" in metadata:
                    print("     Detectado na imagem processada")
                elif "detected_on_original" in metadata:
                    print("     Detectado na imagem original (sem pré-processamento)")
                if "content" in metadata:
                    print(f"     Conteúdo do QR: {metadata['content']}")
                if "preprocessing_variant" in metadata:
                    print(f"     Variante de pré-processamento usada: {metadata['preprocessing_variant']}")
            
            self.update_roi_boxes()
            return True
        except AnchorNotFoundError as e:
            print(f"[ERRO] Âncora não detectada: {str(e)}")
            print("\n[INFO] Dicas:")
            print("     - O QR code pode estar muito granulado ou com baixa qualidade")
            print("     - Tente usar a tecla 'M' para definir a âncora manualmente")
            print("     - Verifique se o QR code está visível na imagem")
            if not self.layout.anchor.try_original_on_failure:
                print("     - Considere habilitar 'try_original_on_failure' no layout para tentar na imagem original")
            return False

    def get_handle_at(self, x: int, y: int, box_idx: int) -> Optional[str]:
        """Verifica se o ponto está em algum handle de redimensionamento."""
        if box_idx < 0 or box_idx >= len(self.roi_boxes):
            return None

        box = self.roi_boxes[box_idx]
        handle_size = 10

        # Cantos
        if abs(x - box["x"]) < handle_size and abs(y - box["y"]) < handle_size:
            return "tl"  # top-left
        if abs(x - (box["x"] + box["w"])) < handle_size and abs(y - box["y"]) < handle_size:
            return "tr"  # top-right
        if abs(x - box["x"]) < handle_size and abs(y - (box["y"] + box["h"])) < handle_size:
            return "bl"  # bottom-left
        if (
            abs(x - (box["x"] + box["w"])) < handle_size
            and abs(y - (box["y"] + box["h"])) < handle_size
        ):
            return "br"  # bottom-right

        # Dentro da caixa (para mover)
        if box["x"] <= x <= box["x"] + box["w"] and box["y"] <= y <= box["y"] + box["h"]:
            return "move"

        return None

    def get_box_at(self, x: int, y: int) -> int:
        """Retorna o índice da caixa que contém o ponto, ou -1."""
        for i, box in enumerate(self.roi_boxes):
            if box["x"] <= x <= box["x"] + box["w"] and box["y"] <= y <= box["y"] + box["h"]:
                return i
        return -1

    def get_viewport_image(self) -> tuple[np.ndarray, tuple[int, int]]:
        """Retorna a imagem do viewport com zoom e pan aplicados."""
        h, w = self.processed_image.shape[:2]

        # Calcular região visível
        zoom_w = int(self.viewport_width / self.zoom_factor)
        zoom_h = int(self.viewport_height / self.zoom_factor)

        # Limitar pan dentro dos limites da imagem
        max_pan_x = max(0, w - zoom_w)
        max_pan_y = max(0, h - zoom_h)
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))

        # Extrair região
        x1 = self.pan_x
        y1 = self.pan_y
        x2 = min(x1 + zoom_w, w)
        y2 = min(y1 + zoom_h, h)

        viewport_img = self.processed_image[y1:y2, x1:x2].copy()

        # Aplicar zoom se necessário
        if self.zoom_factor != 1.0:
            viewport_img = cv2.resize(
                viewport_img, (self.viewport_width, self.viewport_height), interpolation=cv2.INTER_LINEAR
            )

        return viewport_img, (x1, y1)

    def screen_to_image_coords(self, screen_x: int, screen_y: int) -> tuple[int, int]:
        """Converte coordenadas da tela para coordenadas da imagem original."""
        img_x = int(self.pan_x + screen_x / self.zoom_factor)
        img_y = int(self.pan_y + screen_y / self.zoom_factor)
        return img_x, img_y

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Callback para eventos do mouse."""
        # Converter coordenadas da tela para imagem
        img_x, img_y = self.screen_to_image_coords(x, y)

        # Modo de definição manual de âncora
        if self.waiting_for_anchor_click and event == cv2.EVENT_LBUTTONDOWN:
            self.anchor_position = (img_x, img_y)
            self.waiting_for_anchor_click = False
            self.update_roi_boxes()
            self.update_display()
            print(f"[OK] Âncora definida manualmente na posição: {self.anchor_position}")
            return

        if not self.anchor_position:
            return

        if self.creating_new_roi:
            # Modo de criação de novo ROI
            if event == cv2.EVENT_LBUTTONDOWN:
                self.new_roi_start = (img_x, img_y)
            elif event == cv2.EVENT_MOUSEMOVE and self.new_roi_start:
                # Desenhar preview do retângulo
                x1 = min(self.new_roi_start[0], img_x)
                y1 = min(self.new_roi_start[1], img_y)
                x2 = max(self.new_roi_start[0], img_x)
                y2 = max(self.new_roi_start[1], img_y)
                self.new_roi_box = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                self.update_display()
            elif event == cv2.EVENT_LBUTTONUP and self.new_roi_start:
                # Finalizar criação
                x1 = min(self.new_roi_start[0], img_x)
                y1 = min(self.new_roi_start[1], img_y)
                x2 = max(self.new_roi_start[0], img_x)
                y2 = max(self.new_roi_start[1], img_y)

                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:  # Tamanho mínimo
                    self.finalize_new_roi(x1, y1, x2 - x1, y2 - y1)

                self.creating_new_roi = False
                self.new_roi_start = None
                self.new_roi_box = None
                self.update_display()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            # Verificar se clicou em um handle
            if self.current_roi_idx >= 0:
                handle = self.get_handle_at(img_x, img_y, self.current_roi_idx)
                if handle:
                    self.dragging = True
                    self.drag_handle = handle
                    self.drag_start_x = img_x
                    self.drag_start_y = img_y
                    return

            # Verificar se clicou em uma caixa
            box_idx = self.get_box_at(img_x, img_y)
            if box_idx >= 0:
                self.current_roi_idx = box_idx
                self.dragging = True
                self.drag_handle = "move"
                self.drag_start_x = img_x
                self.drag_start_y = img_y
                self.drag_box_start = self.roi_boxes[box_idx].copy()
            else:
                self.current_roi_idx = -1

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.current_roi_idx >= 0:
                box = self.roi_boxes[self.current_roi_idx]
                dx = img_x - self.drag_start_x
                dy = img_y - self.drag_start_y

                if self.drag_handle == "move":
                    box["x"] = self.drag_box_start["x"] + dx
                    box["y"] = self.drag_box_start["y"] + dy
                elif self.drag_handle == "tl":
                    box["x"] = self.drag_box_start["x"] + dx
                    box["y"] = self.drag_box_start["y"] + dy
                    box["w"] = self.drag_box_start["w"] - dx
                    box["h"] = self.drag_box_start["h"] - dy
                elif self.drag_handle == "tr":
                    box["y"] = self.drag_box_start["y"] + dy
                    box["w"] = self.drag_box_start["w"] + dx
                    box["h"] = self.drag_box_start["h"] - dy
                elif self.drag_handle == "bl":
                    box["x"] = self.drag_box_start["x"] + dx
                    box["w"] = self.drag_box_start["w"] - dx
                    box["h"] = self.drag_box_start["h"] + dy
                elif self.drag_handle == "br":
                    box["w"] = self.drag_box_start["w"] + dx
                    box["h"] = self.drag_box_start["h"] + dy

                # Garantir tamanho mínimo
                box["w"] = max(10, box["w"])
                box["h"] = max(10, box["h"])

                # Garantir limites da imagem
                h, w = self.processed_image.shape[:2]
                box["x"] = max(0, min(box["x"], w - box["w"]))
                box["y"] = max(0, min(box["y"], h - box["h"]))

                self.update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            if self.dragging:
                self.dragging = False
                if self.current_roi_idx >= 0:
                    self.update_roi_coords(self.current_roi_idx)
                    self.update_display()

    def update_roi_coords(self, box_idx: int) -> None:
        """Atualiza as coordenadas relativas do ROI baseado na caixa em pixels."""
        if not self.anchor_position or box_idx < 0 or box_idx >= len(self.roi_boxes):
            return

        box = self.roi_boxes[box_idx]
        anchor_x, anchor_y = self.anchor_position

        # Calcular offset relativo à âncora
        offset_x = box["x"] - anchor_x
        offset_y = box["y"] - anchor_y

        # Atualizar ROI no layout
        roi = self.layout.rois[box_idx]
        roi.relative_position = (offset_x, offset_y)
        roi.size = (box["w"], box["h"])

    def finalize_new_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Finaliza a criação de um novo ROI."""
        if not self.anchor_position:
            print("[ERRO] Âncora não definida! Defina a âncora primeiro.")
            return

        anchor_x, anchor_y = self.anchor_position

        # Calcular offset relativo
        offset_x = x - anchor_x
        offset_y = y - anchor_y

        # Pedir informações do ROI
        print("\n" + "=" * 60)
        print("NOVO ROI CRIADO")
        print("=" * 60)
        roi_id = input("ID do ROI (ex: campo_novo): ").strip()
        if not roi_id:
            roi_id = f"roi_{len(self.layout.rois) + 1}"

        roi_label = input("Label do ROI (ex: Nome do Campo): ").strip()
        if not roi_label:
            roi_label = roi_id

        print("\nTipos de OCR disponíveis:")
        print("  1 - text (texto simples)")
        print("  2 - numeric (numérico)")
        print("  3 - date (data)")

        tipo_choice = input("Escolha o tipo (1-3, padrão: 1): ").strip()
        tipo_map = {"1": "text", "2": "numeric", "3": "date"}
        roi_type = tipo_map.get(tipo_choice, "text")

        # Criar novo ROI
        new_roi = ROIConfig(
            id=roi_id,
            label=roi_label,
            relative_position=(offset_x, offset_y),
            size=(w, h),
            ocr_config=OCRConfig(type=roi_type),
            validation=ValidationConfig(required=False),
        )

        # Adicionar ao layout
        self.layout.rois.append(new_roi)

        # Atualizar boxes
        self.update_roi_boxes()

        # Selecionar o novo ROI
        self.current_roi_idx = len(self.roi_boxes) - 1

        print(f"\n[OK] ROI '{roi_id}' criado com sucesso!")
        print(f"     Posição relativa: ({offset_x}, {offset_y})")
        print(f"     Tamanho: {w}x{h}")
        print("=" * 60 + "\n")

    def update_display(self) -> None:
        """Atualiza a imagem exibida com zoom e pan."""
        viewport_img, (offset_x, offset_y) = self.get_viewport_image()
        h_view, w_view = viewport_img.shape[:2]

        # Desenhar âncora se definida
        if self.anchor_position:
            anchor_x = int((self.anchor_position[0] - offset_x) * self.zoom_factor)
            anchor_y = int((self.anchor_position[1] - offset_y) * self.zoom_factor)

            if 0 <= anchor_x < w_view and 0 <= anchor_y < h_view:
                cv2.circle(viewport_img, (anchor_x, anchor_y), 15, (0, 255, 255), 3)
                cv2.line(viewport_img, (anchor_x - 20, anchor_y), (anchor_x + 20, anchor_y), (0, 255, 255), 2)
                cv2.line(viewport_img, (anchor_x, anchor_y - 20), (anchor_x, anchor_y + 20), (0, 255, 255), 2)
                cv2.putText(
                    viewport_img,
                    "ANCORA",
                    (anchor_x + 25, anchor_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # Cores por tipo (cores vibrantes para melhor visibilidade em imagens preto e branco)
        colors = {
            "text": (0, 255, 255),  # Amarelo brilhante
            "numeric": (255, 255, 0),  # Ciano brilhante
            "date": (255, 0, 255),  # Magenta/Rosa brilhante
        }

        # Desenhar ROIs
        for i, box in enumerate(self.roi_boxes):
            roi = box["roi"]
            color = colors.get(roi.ocr_config.type, (255, 255, 255))  # Branco como fallback
            thickness = 5 if i == self.current_roi_idx else 4  # Bordas mais espessas para melhor visibilidade

            # Converter coordenadas da imagem para viewport
            x1 = int((box["x"] - offset_x) * self.zoom_factor)
            y1 = int((box["y"] - offset_y) * self.zoom_factor)
            x2 = int((box["x"] + box["w"] - offset_x) * self.zoom_factor)
            y2 = int((box["y"] + box["h"] - offset_y) * self.zoom_factor)

            # Desenhar apenas se estiver visível no viewport
            if not (x2 < 0 or x1 > w_view or y2 < 0 or y1 > h_view):
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_view, x2)
                y2 = min(h_view, y2)

                cv2.rectangle(viewport_img, (x1, y1), (x2, y2), color, thickness)

                # Handles se selecionado
                if i == self.current_roi_idx:
                    handle_size = int(8 * self.zoom_factor)
                    cv2.circle(viewport_img, (x1, y1), handle_size, (255, 255, 255), -1)
                    cv2.circle(viewport_img, (x2, y1), handle_size, (255, 255, 255), -1)
                    cv2.circle(viewport_img, (x1, y2), handle_size, (255, 255, 255), -1)
                    cv2.circle(viewport_img, (x2, y2), handle_size, (255, 255, 255), -1)

                # Label
                label = f"{i+1}: {roi.label[:30]}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * self.zoom_factor, 1)
                label_y = max(15, y1 - 5)
                cv2.rectangle(
                    viewport_img,
                    (x1, label_y - label_size[1] - 5),
                    (x1 + label_size[0] + 5, label_y + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    viewport_img,
                    label,
                    (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * self.zoom_factor,
                    color,
                    1,
                )

        # Desenhar preview do novo ROI sendo criado
        if self.creating_new_roi and self.new_roi_box:
            x1 = int((self.new_roi_box["x"] - offset_x) * self.zoom_factor)
            y1 = int((self.new_roi_box["y"] - offset_y) * self.zoom_factor)
            x2 = int((self.new_roi_box["x"] + self.new_roi_box["w"] - offset_x) * self.zoom_factor)
            y2 = int((self.new_roi_box["y"] + self.new_roi_box["h"] - offset_y) * self.zoom_factor)

            if not (x2 < 0 or x1 > w_view or y2 < 0 or y1 > h_view):
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w_view, x2)
                y2 = min(h_view, y2)

                cv2.rectangle(viewport_img, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Ciano brilhante para preview, borda mais espessa
                cv2.putText(
                    viewport_img,
                    "NOVO ROI",
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        # Informações de zoom e posição
        info_text = f"Zoom: {self.zoom_factor:.2f}x | Pos: ({self.pan_x}, {self.pan_y})"
        if self.anchor_position:
            info_text += f" | Ancora: {self.anchor_position}"
        cv2.putText(viewport_img, info_text, (10, h_view - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow(self.window_name, viewport_img)

    def save_layout(self) -> None:
        """Salva as alterações no layout JSON."""
        # Criar backup
        backup_path = self.layout_path.with_suffix(".json.backup")
        if not backup_path.exists() and self.layout_path.exists():
            backup_path.write_text(self.layout_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[OK] Backup criado: {backup_path}")

        # Converter layout para dict e salvar
        layout_dict = self.layout.model_dump(mode="json")
        with open(self.layout_path, "w", encoding="utf-8") as f:
            json.dump(layout_dict, f, indent=2, ensure_ascii=False)

        print(f"[OK] Layout salvo: {self.layout_path}")

    def run(self) -> None:
        """Executa a interface interativa."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Definir tamanho da janela
        cv2.resizeWindow(self.window_name, self.viewport_width, self.viewport_height)

        self.update_display()

        print("\n" + "=" * 60)
        print("ROI EDITOR - CONTROLES:")
        print("=" * 60)
        print("  [A] - Detectar âncora automaticamente")
        print("  [M] - Definir âncora manualmente (clique na imagem)")
        print("  Clique em um ROI para selecionar")
        print("  Arraste o ROI para mover")
        print("  Arraste os cantos (círculos brancos) para redimensionar")
        print("  [SETAS] - Navegar entre ROIs")
        print("  [IJKL] - Navegar pela imagem (pan)")
        print("  [+] ou [-] - Zoom in/out")
        print("  [0] - Resetar zoom")
        print("  [N] - Criar novo ROI (arraste para desenhar)")
        print("  [DEL] ou [X] - Deletar ROI atual")
        print("  [S] - Salvar alterações")
        print("  [R] - Resetar ROI atual")
        print("  [Q] ou [ESC] - Sair")
        print("  [H] - Mostrar esta ajuda")
        print("=" * 60 + "\n")

        while True:
            # Usar waitKey com timeout maior para melhor captura de teclas
            key = cv2.waitKey(30) & 0xFF
            
            # Debug: mostrar tecla pressionada (apenas se não for 255 que é "nenhuma tecla")
            if key != 255:
                try:
                    key_char = chr(key) if 32 <= key <= 126 else f"<{key}>"
                    print(f"[DEBUG] Tecla pressionada: '{key_char}' (código: {key})")
                except ValueError:
                    print(f"[DEBUG] Tecla pressionada: <{key}> (código não imprimível)")

            if key == ord("q") or key == 27:  # ESC
                print("[DEBUG] Saindo...")
                break
            elif key == ord("a"):
                print("[DEBUG] Executando detecção de âncora...")
                self.detect_anchor_interactive()
                self.update_display()
            elif key == ord("m"):
                print("[DEBUG] Modo de definição manual de âncora ativado")
                print("\n[INFO] Clique na imagem para definir a posição da âncora")
                self.waiting_for_anchor_click = True
            elif key == ord("s"):
                print("[DEBUG] Salvando layout...")
                self.save_layout()
                print("Alterações salvas!")
            elif key == ord("r") and self.current_roi_idx >= 0:
                print("[DEBUG] Resetando ROI atual...")
                self.update_roi_boxes()
                self.update_display()
                print(f"ROI {self.current_roi_idx + 1} resetado")
            elif key == ord("+") or key == ord("="):
                print("[DEBUG] Zoom in...")
                self.zoom_factor = min(self.zoom_factor * 1.2, 5.0)
                self.update_display()
            elif key == ord("-") or key == ord("_"):
                print("[DEBUG] Zoom out...")
                self.zoom_factor = max(self.zoom_factor / 1.2, 0.1)
                self.update_display()
            elif key == ord("0"):
                print("[DEBUG] Resetando zoom...")
                self.zoom_factor = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self.update_display()
            elif key == ord("i"):
                print("[DEBUG] Pan para cima...")
                self.pan_y = max(0, self.pan_y - int(50 / self.zoom_factor))
                self.update_display()
            elif key == ord("k"):
                print("[DEBUG] Pan para baixo...")
                h, w = self.processed_image.shape[:2]
                self.pan_y = min(h - int(self.viewport_height / self.zoom_factor), self.pan_y + int(50 / self.zoom_factor))
                self.update_display()
            elif key == ord("j"):
                print("[DEBUG] Pan para esquerda...")
                self.pan_x = max(0, self.pan_x - int(50 / self.zoom_factor))
                self.update_display()
            elif key == ord("l"):
                print("[DEBUG] Pan para direita...")
                h, w = self.processed_image.shape[:2]
                self.pan_x = min(w - int(self.viewport_width / self.zoom_factor), self.pan_x + int(50 / self.zoom_factor))
                self.update_display()
            elif key == ord("n"):
                print("[DEBUG] Modo de criação de novo ROI...")
                if not self.anchor_position:
                    print("[ERRO] Defina a âncora primeiro (tecla 'A' ou 'M')!")
                else:
                    self.creating_new_roi = True
                    self.current_roi_idx = -1
                    print("\n[INFO] Modo de criação de ROI ativado!")
                    print("       Clique e arraste na imagem para desenhar o novo ROI")
                    print("       Pressione ESC para cancelar\n")
            elif key == ord("x") or key == 8:  # DEL ou Backspace
                print("[DEBUG] Deletando ROI...")
                if self.current_roi_idx >= 0:
                    roi = self.layout.rois[self.current_roi_idx]
                    confirm = input(f"\nTem certeza que deseja deletar o ROI '{roi.id}'? (s/N): ").strip().lower()
                    if confirm == "s":
                        del self.layout.rois[self.current_roi_idx]
                        self.update_roi_boxes()
                        self.current_roi_idx = -1
                        self.update_display()
                        print(f"[OK] ROI '{roi.id}' deletado!")
                    else:
                        print("Cancelado.")
            elif key == 27 and self.creating_new_roi:  # ESC para cancelar criação
                print("[DEBUG] Cancelando criação de ROI...")
                self.creating_new_roi = False
                self.new_roi_start = None
                self.new_roi_box = None
                self.update_display()
                print("[INFO] Criação de ROI cancelada")
            elif key == ord("h"):
                print("[DEBUG] Mostrando ajuda...")
                print("\nCONTROLES:")
                print("  [A] - Detectar âncora automaticamente")
                print("  [M] - Definir âncora manualmente")
                print("  [N] - Criar novo ROI")
                print("  [X] ou [DEL] - Deletar ROI atual")
                print("  [S] - Salvar alterações")
                print("  [Q] ou [ESC] - Sair\n")
            elif key in [82, 83, 84, 85, 81]:  # Setas
                print(f"[DEBUG] Seta pressionada (código: {key})...")
                if key == 82:  # Seta para cima
                    self.current_roi_idx = (self.current_roi_idx - 1) % len(self.roi_boxes) if self.roi_boxes else -1
                elif key == 84:  # Seta para baixo
                    self.current_roi_idx = (self.current_roi_idx + 1) % len(self.roi_boxes) if self.roi_boxes else -1
                elif key == 83:  # Seta para direita
                    self.current_roi_idx = (self.current_roi_idx + 1) % len(self.roi_boxes) if self.roi_boxes else -1
                elif key == 81:  # Seta para esquerda
                    self.current_roi_idx = (self.current_roi_idx - 1) % len(self.roi_boxes) if self.roi_boxes else -1

                self.update_display()
                if self.current_roi_idx >= 0:
                    roi = self.layout.rois[self.current_roi_idx]
                    print(f"ROI selecionado: {self.current_roi_idx + 1} - {roi.label}")

        cv2.destroyAllWindows()

