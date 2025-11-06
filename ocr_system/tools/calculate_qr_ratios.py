"""Script utilitário para calcular proporções relativas do QR code."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import click
import numpy as np

from ocr_system.core.anchor_detector import detect_qrcode
from ocr_system.core.preprocessing import preprocess_pipeline
from ocr_system.models.anchor import QRCodeAnchorConfig
from ocr_system.models.layout import LayoutConfig
from ocr_system.utils.image_io import load_image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def calculate_qr_ratios(image_path: str | Path, layout_path: str | Path | None = None) -> dict[str, float | int | None]:
    """
    Calcula proporções relativas do QR code a partir de uma imagem.

    Args:
        image_path: Caminho da imagem
        layout_path: Caminho do layout JSON (opcional, para aplicar pré-processamento)

    Returns:
        Dicionário com informações calculadas
    """
    image_path = Path(image_path)
    
    # Carrega imagem
    try:
        image = load_image(image_path)
    except Exception as e:
        raise click.ClickException(f"Erro ao carregar imagem: {e}")
    
    img_height, img_width = image.shape[:2]
    
    # Aplica pré-processamento se layout fornecido
    processed_image = image
    if layout_path:
        try:
            layout_path = Path(layout_path)
            with open(layout_path, "r", encoding="utf-8") as f:
                layout_data = json.load(f)
            layout = LayoutConfig.model_validate(layout_data)
            processed_image, _ = preprocess_pipeline(image, layout.preprocessing, None)
        except Exception as e:
            logger.warning(f"Erro ao aplicar pré-processamento do layout: {e}. Usando imagem original.")
            processed_image = image
    
    # Detecta QR code
    config = QRCodeAnchorConfig()
    result = detect_qrcode(processed_image, config)
    
    if result is None:
        raise click.ClickException(
            "QR code não detectado na imagem. Verifique se há um QR code visível."
        )
    
    position, confidence, metadata = result
    qr_width = metadata.get("qr_width")
    qr_height = metadata.get("qr_height")
    
    if qr_width is None or qr_height is None:
        raise click.ClickException("Não foi possível obter dimensões do QR code detectado.")
    
    # Calcula proporções
    width_ratio = qr_width / img_width
    height_ratio = qr_height / img_height
    
    return {
        "image_width": img_width,
        "image_height": img_height,
        "qr_width": qr_width,
        "qr_height": qr_height,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "qr_position": position,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "--image",
    required=True,
    type=click.Path(exists=True),
    help="Caminho da imagem com QR code"
)
@click.option(
    "--layout",
    type=click.Path(exists=True),
    help="Caminho do layout JSON (opcional, para aplicar pré-processamento)"
)
@click.option(
    "--output",
    type=click.Path(),
    help="Arquivo JSON para salvar configuração calculada (opcional)"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "inline", "pretty"], case_sensitive=False),
    default="pretty",
    help="Formato de saída: json, inline ou pretty (padrão)"
)
def main(image: str, layout: str | None, output: str | None, output_format: str) -> None:
    """
    Calcula proporções relativas do QR code a partir de uma imagem.

    Exemplos:

    \b
    # Calcular proporções de uma imagem
    python -m ocr_system.tools.calculate_qr_ratios --image docs/page1.jpg

    \b
    # Calcular com pré-processamento do layout
    python -m ocr_system.tools.calculate_qr_ratios --image docs/page1.jpg --layout configs/layouts/smart_v1.json

    \b
    # Salvar configuração em arquivo JSON
    python -m ocr_system.tools.calculate_qr_ratios --image docs/page1.jpg --output qr_config.json
    """
    try:
        result = calculate_qr_ratios(image, layout)
        
        # Formata saída
        if output_format == "json":
            output_data = {
                "expected_qr_width_ratio": round(result["width_ratio"], 6),
                "expected_qr_height_ratio": round(result["height_ratio"], 6),
                "expected_qr_size": [result["qr_width"], result["qr_height"]],
            }
            output_str = json.dumps(output_data, indent=2)
        elif output_format == "inline":
            output_str = (
                f'expected_qr_width_ratio: {result["width_ratio"]:.6f}, '
                f'expected_qr_height_ratio: {result["height_ratio"]:.6f}'
            )
        else:  # pretty
            click.echo("\n" + "=" * 60)
            click.echo("📊 CÁLCULO DE PROPORÇÕES DO QR CODE")
            click.echo("=" * 60)
            click.echo(f"\n📷 Imagem: {Path(image).name}")
            click.echo(f"   Dimensões: {result['image_width']} x {result['image_height']} pixels")
            click.echo(f"\n🔲 QR Code Detectado:")
            click.echo(f"   Tamanho: {result['qr_width']} x {result['qr_height']} pixels")
            click.echo(f"   Posição: {result['qr_position']}")
            click.echo(f"   Confiança: {result['confidence']:.2f}")
            click.echo(f"\n📐 Proporções Relativas:")
            click.echo(f"   Largura: {result['width_ratio']:.6f} ({result['width_ratio']*100:.2f}%)")
            click.echo(f"   Altura:  {result['height_ratio']:.6f} ({result['height_ratio']*100:.2f}%)")
            click.echo(f"\n📝 Configuração JSON:")
            click.echo("-" * 60)
            config_json = json.dumps({
                "expected_qr_width_ratio": round(result["width_ratio"], 6),
                "expected_qr_height_ratio": round(result["height_ratio"], 6),
                "expected_qr_size": [result["qr_width"], result["qr_height"]],
            }, indent=2)
            click.echo(config_json)
            click.echo("-" * 60)
            click.echo("\n💡 Dica: Adicione esses campos na seção 'primary' do seu layout JSON")
            output_str = config_json
        
        # Salva em arquivo se solicitado
        if output:
            output_path = Path(output)
            if output_format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(output_str)
            else:
                # Salva apenas a parte JSON
                output_data = {
                    "expected_qr_width_ratio": round(result["width_ratio"], 6),
                    "expected_qr_height_ratio": round(result["height_ratio"], 6),
                    "expected_qr_size": [result["qr_width"], result["qr_height"]],
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2)
            click.echo(f"\n✅ Configuração salva em: {output_path}")
        
    except Exception as e:
        raise click.ClickException(f"Erro: {e}")


if __name__ == "__main__":
    main()

