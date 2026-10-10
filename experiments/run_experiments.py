"""
Modulo para la ejecucion automatizada de la matriz de experimentos de compresion Quadtree.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Permitir importar modulos desde el directorio raiz (src)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.quadtree import build_quadtree, reconstruct_image, get_tree_stats
from src.metrics import calculate_mse, calculate_bit_size, calculate_compression_ratio


def generate_synthetic_images(output_dir="images"):
    """
    Genera las imagenes de prueba en tres dimensiones y tres categorias de contenido:
    1. Uniforme (Color plano)
    2. Formas definidas (Regiones geometricas)
    3. Compleja / Detalles (Gradientes con textura y ruido)
    """
    os.makedirs(output_dir, exist_ok=True)
    sizes = [64, 128, 256]
    generated_files = []

    for size in sizes:
        # 1. Imagen Uniforme
        img_uniform = np.full((size, size, 3), [120, 180, 200], dtype=np.uint8)
        path_u = os.path.join(output_dir, f"uniforme_{size}x{size}.png")
        Image.fromarray(img_uniform).save(path_u)
        generated_files.append((f"Uniforme {size}x{size}", path_u, size, size))

        # 2. Imagen con Formas Definidas
        pil_shapes = Image.new("RGB", (size, size), color=(240, 240, 240))
        draw = ImageDraw.Draw(pil_shapes)
        draw.rectangle([size // 8, size // 8, size // 2, size // 2], fill=(220, 50, 50))
        draw.ellipse([size // 2, size // 2, size - size // 8, size - size // 8], fill=(50, 180, 50))
        draw.rectangle([size // 2, size // 8, size - size // 8, size // 2], fill=(50, 50, 220))
        path_s = os.path.join(output_dir, f"formas_{size}x{size}.png")
        pil_shapes.save(path_s)
        generated_files.append((f"Formas {size}x{size}", path_s, size, size))

        # 3. Imagen Compleja / Textura con Ruido
        x_mesh = np.linspace(0, 1, size)
        y_mesh = np.linspace(0, 1, size)
        xx, yy = np.meshgrid(x_mesh, y_mesh)
        r_ch = (np.sin(xx * 10) * 127 + 128).astype(np.uint8)
        g_ch = (np.cos(yy * 10) * 127 + 128).astype(np.uint8)
        b_ch = ((xx + yy) * 127).astype(np.uint8)
        img_complex = np.stack([r_ch, g_ch, b_ch], axis=-1)
        
        noise = np.random.randint(-20, 20, (size, size, 3), dtype=np.int16)
        img_complex_noisy = np.clip(img_complex.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        path_c = os.path.join(output_dir, f"compleja_{size}x{size}.png")
        Image.fromarray(img_complex_noisy).save(path_c)
        generated_files.append((f"Compleja {size}x{size}", path_c, size, size))

    return generated_files


def save_visual_evidence(original, reconstructed, exp_id, name, tolerance, output_dir="evidence"):
    """
    Guarda una imagen comparativa (Original vs Reconstruida) para el Entregable 4.
    """
    os.makedirs(output_dir, exist_ok=True)
    _, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(reconstructed)
    axes[1].set_title(f"Reconstruida (T={tolerance})")
    axes[1].axis("off")

    plt.suptitle(f"Experimento {exp_id}: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_id}_{name}_T{tolerance}.png"), dpi=150)
    plt.close()


def run_all_experiments():
    """
    Ejecuta la matriz de pruebas completa registrando las metricas obligatorias.
    """
    print("=== Iniciando Generacion de Imagenes de Prueba ===")
    test_images = generate_synthetic_images()

    tolerances = [0, 15, 40]
    results = []
    exp_counter = 1

    print("\n=== Ejecutando Matriz de Experimentos ===")
    for label, img_path, width, height in test_images:
        original_img = np.array(Image.open(img_path).convert("RGB"))

        for tol in tolerances:
            exp_id = f"EXP_{exp_counter:02d}"

            start_time = time.perf_counter()
            root_node = build_quadtree(original_img, 0, 0, width, height, tolerance=tol)
            construction_time_ms = (time.perf_counter() - start_time) * 1000.0

            stats = get_tree_stats(root_node)

            reconstructed_img = np.zeros_like(original_img)
            reconstruct_image(root_node, reconstructed_img)

            mse = calculate_mse(original_img, reconstructed_img)
            orig_bits, comp_bits = calculate_bit_size(stats, width, height)
            cr = calculate_compression_ratio(orig_bits, comp_bits)

            save_visual_evidence(original_img, reconstructed_img, exp_id, label.replace(" ", "_"), tol)

            record = {
                "ID_Exp": exp_id,
                "Tipo_Imagen": label,
                "Ancho": width,
                "Alto": height,
                "Tolerancia": tol,
                "Tiempo_Construccion_ms": round(construction_time_ms, 3),
                "Total_Nodos": stats["total_nodes"],
                "Nodos_Hoja": stats["leaf_nodes"],
                "Profundidad_Max": stats["max_depth"],
                "Tam_Original_bits": orig_bits,
                "Tam_Comprimido_bits": comp_bits,
                "Razon_Compresion_CR": round(cr, 2),
                "Error_MSE": round(mse, 2)
            }
            results.append(record)

            print(f"[{exp_id}] {label} | T={tol:2d} | Nodos: {stats['total_nodes']:5d} | CR: {cr:6.2f}x | MSE: {mse:6.2f} | T_ms: {construction_time_ms:6.2f}")
            exp_counter += 1

    df_results = pd.DataFrame(results)
    os.makedirs("experiments", exist_ok=True)
    csv_path = os.path.join("experiments", "results.csv")
    df_results.to_csv(csv_path, index=False)

    print("\n==========================================")
    print("¡Matriz de experimentos completada con exito!")
    print(f"Resultados guardados en: {csv_path}")
    print("Evidencias visuales guardadas en: evidence/")
    print("==========================================")


if __name__ == "__main__":
    run_all_experiments()
    