import numpy as np

class QuadTreeNode:
    """
    Estructura de nodo para un Quadtree de compresión de imágenes.
    """
    def __init__(self, x, y, width, height, is_leaf=False, color=None):
        self.x = x            # Coordenada X del borde superior izquierdo
        self.y = y            # Coordenada Y del borde superior izquierdo
        self.width = width    # Ancho de la región
        self.height = height  # Alto de la región
        self.is_leaf = is_leaf  # True si es nodo hoja
        self.color = color    # Tupla (R, G, B) si es hoja, None si es nodo interno
        
        # Punteros a subregiones
        self.nw = None  # Noroeste
        self.ne = None  # Noreste
        self.sw = None  # Suroeste
        self.se = None  # Sureste


def calculate_region_color_and_error(image, x, y, width, height):
    """
    Calcula el color medio (RGB) y el error (desviación estándar RMS) de una región.
    """
    region = image[y:y+height, x:x+width]
    
    # Manejo de bordes o regiones vacías
    if region.size == 0:
        return (0, 0, 0), 0.0

    # Color medio por canal RGB
    mean_color = np.mean(region, axis=(0, 1))
    
    # Desviación estándar por canal
    std_channels = np.std(region, axis=(0, 1))
    
    # Desviación RMS agregada entre los 3 canales
    error = np.sqrt(np.mean(std_channels ** 2))
    
    return tuple(mean_color.astype(int)), float(error)


def build_quadtree(image, x, y, width, height, tolerance):
    """
    Construye recursivamente el Quadtree aplicando la estrategia Divide y Vencerás.
    
    Estrategia:
    - DIVIDIR: Particionar la región [width x height] en 4 cuadrantes.
    - RESOLVER: Evaluar condición base (uniformidad <= tolerancia) o procesar subregiones.
    - COMBINAR: Retornar nodo interno enlazando las 4 subregiones.
    """
    if width <= 0 or height <= 0:
        return None

    mean_color, error = calculate_region_color_and_error(image, x, y, width, height)

    # CASO BASE: La región cumple el criterio de tolerancia o es de 1x1 píxel
    if error <= tolerance or (width == 1 and height == 1):
        return QuadTreeNode(x, y, width, height, is_leaf=True, color=mean_color)

    # DIVIDIR: Calcular puntos medios (soporta dimensiones no potencias de 2)
    mid_w = width // 2
    mid_h = height // 2

    node = QuadTreeNode(x, y, width, height, is_leaf=False, color=None)

    # RESOLVER: Llamadas recursivas para cada cuadrante
    node.nw = build_quadtree(image, x, y, mid_w, height - mid_h, tolerance)
    node.ne = build_quadtree(image, x + mid_w, y, width - mid_w, height - mid_h, tolerance)
    node.sw = build_quadtree(image, x, y + mid_h, mid_w, mid_h, tolerance)
    node.se = build_quadtree(image, x + mid_w, y + mid_h, width - mid_w, mid_h, tolerance)

    # COMBINAR: Retorna el nodo raíz de este subárbol
    return node


def reconstruct_image(node, output_image):
    """
    Reconstruye la matriz de imagen a partir del recorrido DFS del Quadtree.
    """
    if node is None:
        return

    if node.is_leaf:
        # Asignar el color de la hoja a todos los píxeles de su región
        output_image[node.y:node.y+node.height, node.x:node.x+node.width] = node.color
    else:
        # Recorrido recursivo en profundidad
        reconstruct_image(node.nw, output_image)
        reconstruct_image(node.ne, output_image)
        reconstruct_image(node.sw, output_image)
        reconstruct_image(node.se, output_image)


def get_tree_stats(node):
    """
    Obtiene métricas estructurales del árbol: total nodos, nodos hoja y profundidad máxima.
    """
    if node is None:
        return {"total_nodes": 0, "leaf_nodes": 0, "max_depth": 0}

    if node.is_leaf:
        return {"total_nodes": 1, "leaf_nodes": 1, "max_depth": 1}

    nw_s = get_tree_stats(node.nw)
    ne_s = get_tree_stats(node.ne)
    sw_s = get_tree_stats(node.sw)
    se_s = get_tree_stats(node.se)

    total_nodes = 1 + nw_s["total_nodes"] + ne_s["total_nodes"] + sw_s["total_nodes"] + se_s["total_nodes"]
    leaf_nodes = nw_s["leaf_nodes"] + ne_s["leaf_nodes"] + sw_s["leaf_nodes"] + se_s["leaf_nodes"]
    max_depth = 1 + max(nw_s["max_depth"], ne_s["max_depth"], sw_s["max_depth"], se_s["max_depth"])

    return {
        "total_nodes": total_nodes,
        "leaf_nodes": leaf_nodes,
        "max_depth": max_depth
    }