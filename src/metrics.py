import numpy as np

def calculate_mse(original_image, reconstructed_image):
    """
    Calcula el Error Cuadrático Medio (MSE) entre la imagen original y la reconstruida.
    """
    orig = original_image.astype(np.float64)
    recon = reconstructed_image.astype(np.float64)
    mse = np.mean((orig - recon) ** 2)
    return float(mse)


def calculate_bit_size(tree_stats, img_width, img_height):
    """
    Calcula la representación estimada en bits de la estructura Quadtree.
    
    Formato de representación binaria:
    - Metadatos del encabezado: 32 bits (16 bits para Ancho + 16 bits para Alto)
    - Cada Nodo Interno: 1 bit ('0')
    - Cada Nodo Hoja: 1 bit ('1') + 24 bits (Color RGB de 8 bits por canal)
    """
    leaf_nodes = tree_stats["leaf_nodes"]
    internal_nodes = tree_stats["total_nodes"] - leaf_nodes

    # Metadatos
    header_bits = 32

    # Nodos
    internal_bits = internal_nodes * 1
    leaf_bits = leaf_nodes * (1 + 24)

    total_compressed_bits = header_bits + internal_bits + leaf_bits
    original_bits = img_width * img_height * 24

    return original_bits, total_compressed_bits


def calculate_compression_ratio(original_bits, compressed_bits):
    """
    Calcula la Razón de Compresión (Compression Ratio - CR).
    CR = Tamaño Original (bits) / Tamaño Comprimido (bits)
    """
    if compressed_bits == 0:
        return 0.0
    return float(original_bits / compressed_bits)