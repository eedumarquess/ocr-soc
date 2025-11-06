"""CLI para o ROI Editor."""

import sys
from pathlib import Path

import click

from ocr_system.core.processor import load_layout
from ocr_system.core.preprocessing import preprocess_pipeline
from ocr_system.tools.roi_editor import ROISelector
from ocr_system.utils.image_io import load_image


@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.argument("layout_id", type=str)
@click.option("--layouts-dir", type=click.Path(exists=True), help="Diretório de layouts (opcional)")
@click.option("--preprocess", is_flag=True, help="Aplicar pré-processamento antes de editar")
def edit_rois(image_path: str, layout_id: str, layouts_dir: str | None, preprocess: bool) -> None:
    """
    Abre o editor interativo de ROIs.

    Exemplo:
        python -m ocr_system.cli.roi_editor docs/page1.jpg page1_qr_test --preprocess
    """
    image_path = Path(image_path)
    layouts_path = Path(layouts_dir) if layouts_dir else None

    try:
        # Carregar layout
        layout = load_layout(layout_id, layouts_path)
        layout_path = (layouts_path or Path(__file__).parent.parent.parent / "configs" / "layouts") / f"{layout_id}.json"

        if not layout_path.exists():
            click.echo(f"❌ Layout não encontrado: {layout_path}", err=True)
            sys.exit(1)

        # Carregar imagem
        image = load_image(image_path)
        click.echo(f"✅ Imagem carregada: {image_path} ({image.shape[1]}x{image.shape[0]})")

        # Pré-processar se solicitado
        processed_image = image
        if preprocess:
            click.echo("\n[INFO] Aplicando pré-processamento...")
            processed_image, _ = preprocess_pipeline(image, layout.preprocessing)
            click.echo(f"✅ Imagem processada: {processed_image.shape[1]}x{processed_image.shape[0]}")

        # Criar editor
        selector = ROISelector(image, layout, layout_path, processed_image=processed_image)

        # Tentar detectar âncora automaticamente
        click.echo("\n[INFO] Tentando detectar âncora automaticamente...")
        selector.detect_anchor_interactive()

        # Executar editor
        selector.run()

        click.echo("\n✅ Finalizado!")

    except FileNotFoundError as e:
        click.echo(f"❌ Erro: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Erro inesperado: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    edit_rois()


