"""CLI principal para processamento de documentos."""

import json
import logging
from pathlib import Path
from typing import Any

import click

from ocr_system.core.processor import format_output_json, load_layout, process_document
from ocr_system.utils.visualization import save_roi_visualization


@click.group()
def cli() -> None:
    """Sistema OCR com âncoras para laudos médicos."""
    pass


@cli.command()
@click.option("--image", required=True, type=click.Path(exists=True), help="Caminho da imagem")
@click.option("--layout", required=True, help="ID do layout (nome do arquivo sem .json)")
@click.option("--output", type=click.Path(), help="Arquivo JSON de saída (opcional)")
@click.option("--debug", is_flag=True, help="Salvar imagens intermediárias")
@click.option("--visualize", is_flag=True, help="Salvar imagem com ROIs desenhados")
@click.option("--layouts-dir", type=click.Path(exists=True), help="Diretório de layouts (opcional)")
@click.option("--verbose", "-v", is_flag=True, help="Exibir logs detalhados da extração")
def process(image: str, layout: str, output: str | None, debug: bool, visualize: bool, layouts_dir: str | None, verbose: bool) -> None:
    """Processa um único documento."""
    # Configura logging
    if verbose:
        # Força reconfiguração dos loggers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # Remove handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        # Adiciona novo handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root_logger.addHandler(handler)
        # Configura loggers específicos
        for logger_name in ['ocr_system.core.ocr_engine', 'ocr_system.core.roi_extractor']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.propagate = True  # Propaga para root logger
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
    
    image_path = Path(image)
    layouts_path = Path(layouts_dir) if layouts_dir else None

    try:
        # Carrega layout
        layout_config = load_layout(layout, layouts_path)

        # Diretório de debug
        debug_dir = None
        if debug:
            debug_dir = image_path.parent / f"{image_path.stem}_debug"
            debug_dir.mkdir(exist_ok=True)

        # Processa documento
        result = process_document(image_path, layout_config, debug_dir)

        # Salva visualização se solicitado
        if visualize:
            vis_path = image_path.parent / f"{image_path.stem}_rois.jpg"
            from ocr_system.utils.image_io import load_image
            processed_img = load_image(image_path)
            # Aplicar pré-processamento se necessário
            from ocr_system.core.preprocessing import preprocess_pipeline
            processed_img, _ = preprocess_pipeline(processed_img, layout_config.preprocessing)
            save_roi_visualization(processed_img, result, layout_config, vis_path, show_values=True)
            click.echo(f"[OK] Visualização salva em: {vis_path}")

        # Formata e salva ou exibe resultado em JSON amigável
        output_data = format_output_json(result, layout_config)

        if output:
            output_path = Path(output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            click.echo(f"Resultado salvo em: {output_path}")
        else:
            click.echo(json.dumps(output_data, indent=2, ensure_ascii=False))

        # Exibe estatísticas
        if result.errors:
            click.echo(f"\n[AVISO] {len(result.errors)} erro(s) encontrado(s):", err=True)
            for error in result.errors:
                click.echo(f"  - {error}", err=True)

        click.echo(f"\n[OK] Taxa de sucesso: {result.success_rate:.1%}")
        click.echo(f"   Ancora detectada: {'Sim' if result.anchor_detected else 'Nao'}")
        click.echo(f"   Campos extraidos: {len([f for f in result.fields if f.value])}/{len(result.fields)}")

    except FileNotFoundError as e:
        click.echo(f"[ERRO] {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"[ERRO] Erro inesperado: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True), help="Diretório com imagens")
@click.option("--layout", required=True, help="ID do layout")
@click.option("--output", required=True, type=click.Path(), help="Arquivo JSON de saída")
@click.option("--pattern", default="*.jpg", help="Padrão de arquivos (ex: *.jpg, *.png)")
@click.option("--layouts-dir", type=click.Path(exists=True), help="Diretório de layouts (opcional)")
@click.option("--debug", is_flag=True, help="Salvar imagens intermediárias")
@click.option("--verbose", "-v", is_flag=True, help="Exibir logs detalhados da extração")
def batch(
    input_dir: str,
    layout: str,
    output: str,
    pattern: str,
    layouts_dir: str | None,
    debug: bool,
    verbose: bool,
) -> None:
    """Processa lote de documentos."""
    # Configura logging
    if verbose:
        # Força reconfiguração dos loggers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # Remove handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        # Adiciona novo handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root_logger.addHandler(handler)
        # Configura loggers específicos
        for logger_name in ['ocr_system.core.ocr_engine', 'ocr_system.core.roi_extractor']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.propagate = True  # Propaga para root logger
    else:
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
    
    input_path = Path(input_dir)
    layouts_path = Path(layouts_dir) if layouts_dir else None

    try:
        # Carrega layout
        layout_config = load_layout(layout, layouts_path)

        # Encontra imagens
        image_files = list(input_path.glob(pattern))
        if not image_files:
            click.echo(f"[ERRO] Nenhuma imagem encontrada com padrao '{pattern}' em {input_path}", err=True)
            raise click.Abort()

        click.echo(f"[INFO] Processando {len(image_files)} documento(s)...")

        # Processa cada imagem
        results: list[dict[str, Any]] = []
        for i, image_path in enumerate(image_files, 1):
            click.echo(f"\n[{i}/{len(image_files)}] Processando: {image_path.name}")

            debug_dir = None
            if debug:
                debug_dir = input_path / f"{image_path.stem}_debug"
                debug_dir.mkdir(exist_ok=True)

            try:
                result = process_document(image_path, layout_config, debug_dir)
                results.append(format_output_json(result, layout_config))

                # Feedback rápido
                status = "[OK]" if result.anchor_detected and result.success_rate > 0.8 else "[AVISO]"
                click.echo(f"   {status} Sucesso: {result.success_rate:.1%}")

            except Exception as e:
                click.echo(f"   [ERRO] {e}", err=True)
                results.append(
                    {
                        "layout_id": layout_config.layout_id,
                        "image_path": str(image_path),
                        "anchor_detected": False,
                        "errors": [str(e)],
                        "fields": [],
                    }
                )

        # Salva resultados
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        click.echo(f"\n[OK] Resultados salvos em: {output_path}")
        click.echo(f"   Total processado: {len(results)}")

    except FileNotFoundError as e:
        click.echo(f"[ERRO] {e}", err=True)
        raise click.Abort()
    except Exception as e:
        click.echo(f"[ERRO] Erro inesperado: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
