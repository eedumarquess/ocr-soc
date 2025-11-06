# Sistema OCR com Âncoras e Layouts Reutilizáveis

Sistema modular de OCR para laudos médicos padronizados onde layouts são configurados uma única vez e reutilizados em múltiplos documentos do mesmo tipo.

## Características

- **Calibração única**: Configure o layout uma vez, use em múltiplos documentos
- **Sistema de âncoras**: Compensa variações de escaneamento usando QR codes, texto ou formas
- **Coordenadas relativas**: ROIs calculados dinamicamente a partir de pontos de referência
- **Pré-processamento configurável**: Deskew, binarização, denoise, normalização de contraste
- **Validação robusta**: Regex, ranges e validações customizadas por campo

## Instalação

```bash
pip install -e .
```

## Uso

### Criar/Editar Layouts (ROI Editor)

Use o editor interativo para criar e ajustar ROIs:

```bash
python -m ocr_system.cli.roi_editor docs/page1.jpg page1_qr_test --preprocess
```

**Controles do Editor:**
- `A` - Detectar âncora automaticamente
- `M` - Definir âncora manualmente (clique na imagem)
- `N` - Criar novo ROI (arraste para desenhar)
- Clique e arraste - Mover ROI
- Arraste cantos - Redimensionar ROI
- `S` - Salvar alterações
- `Q` ou `ESC` - Sair

### Processar um único documento

```bash
python -m ocr_system.cli.main process --image laudo_001.jpg --layout lab_hemograma
```

### Processar lote

```bash
python -m ocr_system.cli.main batch --input-dir ./scans/ --layout lab_hemograma --output results.json
```

## Estrutura

```
ocr_system/
├── cli/          # Interface de linha de comando
├── core/         # Módulos principais (preprocessing, anchors, OCR)
├── models/       # Schemas Pydantic
└── utils/        # Utilitários (I/O, normalizações, geometria)
```

## Desenvolvimento

```bash
pip install -e ".[dev]"
pytest
```

