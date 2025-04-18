import sys
import os
import torch
import json
import numpy as np
from PIL import Image, ImageOps
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem, QHBoxLayout, QLineEdit, QSplitter, QProgressBar, QSizePolicy, QTabWidget, QStyleFactory, QMessageBox, QDialog)
from PyQt6.QtGui import QPixmap, QIcon, QDrag, QFontDatabase, QFont, QPalette, QColor, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QMimeData, QEventLoop
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions, overlay_colored_masks
from datetime import datetime
import re
import cv2
from scipy.spatial.distance import pdist, squareform
import tifffile
import xml.etree.ElementTree as ET
from skimage import measure        
import csv
from skimage.measure import regionprops
from scipy import ndimage

SETTINGS_FILE = "settings.json"

# Estilos para la aplicación
def get_app_styles(is_dark_mode):
    # Colores para modo oscuro
    if is_dark_mode:
        primary_color = "#0078D7"  # Azul eléctrico
        accent_color = "#00BFFF"   # Azul brillante
        bg_color = "#1E1E1E"       # Fondo oscuro
        card_bg = "#2D2D30"        # Fondo de tarjetas
        text_color = "#FFFFFF"     # Texto blanco
        secondary_text = "#CCCCCC" # Texto secundario
        border_color = "#333333"   # Bordes
    else:
        # Colores para modo claro
        primary_color = "#007ACC"  # Azul más suave
        accent_color = "#0066CC"   # Azul acento
        bg_color = "#F8F8F8"       # Fondo claro
        card_bg = "#FFFFFF"        # Fondo de tarjetas
        text_color = "#212121"     # Texto oscuro
        secondary_text = "#757575" # Texto secundario
        border_color = "#DDDDDD"   # Bordes
    
    return f"""
        QWidget {{
            background-color: {bg_color};
            color: {text_color};
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        
        QLabel {{
            color: {text_color};
        }}
        
        QTabWidget::pane {{
            border: 1px solid {border_color};
            background-color: {card_bg};
            border-radius: 4px;
        }}
        
        QTabWidget::tab-bar {{
            alignment: center;
        }}
        
        QTabBar::tab {{
            background-color: {card_bg};
            color: {secondary_text};
            padding: 8px 12px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        
        QTabBar::tab:selected {{
            background-color: {primary_color};
            color: white;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {accent_color};
            color: white;
        }}
        
        QPushButton {{
            background-color: {primary_color};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
            font-weight: 500;
        }}
        
        QPushButton:hover {{
            background-color: {accent_color};
        }}
        
        QPushButton:pressed {{
            background-color: {primary_color};
        }}
        
        QPushButton:disabled {{
            background-color: {border_color};
            color: {secondary_text};
        }}
        
        QLineEdit {{
            border: 1px solid {border_color};
            border-radius: 4px;
            padding: 8px;
            background-color: {card_bg};
            selection-background-color: {primary_color};
        }}
        
        QProgressBar {{
            border: 1px solid {border_color};
            border-radius: 4px;
            background-color: {card_bg};
            text-align: center;
            color: {text_color};
        }}
        
        QProgressBar::chunk {{
            background-color: {primary_color};
            border-radius: 3px;
        }}
        
        QListWidget {{
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: 4px;
            padding: 5px;
        }}
        
        QListWidget::item:selected {{
            background-color: {primary_color};
            color: white;
            border-radius: 2px;
        }}
        
        QListWidget::item:hover:!selected {{
            background-color: {accent_color}80;  /* 50% transparent */
            border-radius: 2px;
        }}
        
        QSplitter::handle {{
            background-color: {border_color};
        }}
        
        QSplitter::handle:horizontal {{
            width: 1px;
        }}
        
        QSplitter::handle:hover {{
            background-color: {accent_color};
        }}
    """

class ImageProcessorThread(QThread):
    progress_updated = pyqtSignal(int, str)
    processing_done = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_folder, api_key, project_id):
        super().__init__()
        self.image_folder = image_folder
        self.api_key = api_key
        self.project_id = project_id
        self.should_stop = False
        self.predictor = None
    
    def stop(self):
        self.should_stop = True
    
    def run(self):
        """Método principal que ejecuta el procesamiento de imágenes."""
        try:
            # Configurar directorios y archivos
            timestamp = self.setup_directories()
            
            # Inicializar archivos CSV
            detailed_csv_path, summary_csv_path = self.setup_csv_files(timestamp)
            
            # Inicializar predictor
            if not self.initialize_predictor():
                return
            
            # Obtener la lista de imágenes
            images = self.get_image_files()
            
            # Procesar cada imagen
            for i, filename in enumerate(images, start=1):
                if self.should_stop:
                    break
                
                try:
                    self.process_single_image(filename, i, detailed_csv_path, summary_csv_path, timestamp)
                except Exception as e:
                    self.error_occurred.emit(f"Error procesando imagen {filename}: {str(e)}")
                    continue
            
            self.processing_done.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Error general: {str(e)}")
    
    def setup_directories(self):
        """Crea la estructura de directorios para los resultados."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_folder = os.path.join(self.image_folder, f"processed_images_{timestamp}")
        
        # Crear estructura de directorios
        os.makedirs(main_folder, exist_ok=True)
        
        # Crear subcarpetas
        for subfolder in ["overlayed_images", "masks", "masks_calculations"]:
            os.makedirs(os.path.join(main_folder, subfolder), exist_ok=True)
        
        return timestamp
    
    def setup_csv_files(self, timestamp):
        """Inicializa los archivos CSV con sus encabezados."""
        import csv
        
        main_folder = os.path.join(self.image_folder, f"processed_images_{timestamp}")
        
        # Paths de los archivos CSV
        detailed_csv_path = os.path.join(main_folder, "detailed_results.csv")
        summary_csv_path = os.path.join(main_folder, "summary_results.csv")
        
        # Inicializar CSV detallado - Añadir columna para diámetro del círculo máximo
        with open(detailed_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Imagen', 'Número de objeto', 'Diámetro máximo de Feret (μm)', 
                             'Diámetro del círculo máximo (μm)', 'Circularidad'])
        
        # Inicializar CSV de resumen - Añadir columna para diámetro promedio del círculo máximo
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Imagen', 'Número de objetos', 'Promedio diámetro de Feret (μm)', 
                             'Promedio diámetro del círculo máximo (μm)', 'Promedio circularidad'])
        
        return detailed_csv_path, summary_csv_path
    
    def initialize_predictor(self):
        """Inicializa el predictor de LandingAI."""
        try:
            from landingai.predict import Predictor
            self.predictor = Predictor(endpoint_id=self.project_id, api_key=self.api_key)
            return True
        except ConnectionError as e:
            self.error_occurred.emit("Error de conexión con LandingAI. Comprueba tu conexión a internet y que la API Key sea correcta.")
            return False
        except Exception as e:
            self.error_occurred.emit(f"Error al inicializar el predictor: {str(e)}")
            return False
    
    def get_image_files(self):
        """Obtiene la lista de archivos de imagen en la carpeta."""
        return [f for f in os.listdir(self.image_folder) 
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif"))]
    
    def process_single_image(self, filename, index, detailed_csv_path, summary_csv_path, timestamp):
        """Procesa una imagen individual."""
        import cv2
        import numpy as np
        from PIL import Image
        from skimage.measure import regionprops
        from landingai.visualize import overlay_predictions, overlay_colored_masks
        
        # Paths relevantes
        image_path = os.path.join(self.image_folder, filename)
        main_folder = os.path.join(self.image_folder, f"processed_images_{timestamp}")
        base_name = os.path.splitext(filename)[0]
        
        # Obtener micrómetros por píxel
        micrometers_per_pixel = self.extract_microns_per_pixel(image_path, filename)
        
        # Cargar y procesar imagen
        img = Image.open(image_path).convert("RGB")
        predictions = self.predictor.predict(img)

        # Guardar imagen con overlay en formato JPG
        overlayed_img = overlay_predictions(predictions, img)
        # Cambiar extensión a .jpg
        overlay_filename_jpg = f"{base_name}.jpg"
        overlay_path = os.path.join(main_folder, "overlayed_images", overlay_filename_jpg)
        # Guardar con formato JPEG
        overlayed_img.save(overlay_path, format='JPEG', quality=90) # Calidad 90 (ajustable)

        # solo si hay predicciones, creamos la máscara y el resto de cosas
        if predictions:
        
            # Crear y guardar máscara básica
            mask_img, mask_path = self.create_basic_mask(img, predictions, main_folder, base_name, ".jpg") # Pasar extensión .jpg
            
            # Procesar máscara para análisis
            clean_mask = self.process_mask(mask_img, main_folder, base_name, ".jpg") # Pasar extensión .jpg
            
            # Analizar objetos en la máscara
            object_data, feret_diameters, circle_diameters, circularity_values = self.analyze_objects(
                clean_mask, micrometers_per_pixel, detailed_csv_path, filename)
            
            # Escribir resumen en CSV
            self.write_summary_csv(
                summary_csv_path, filename, object_data, feret_diameters, circle_diameters, circularity_values)
            
            # Crear visualización con datos
            self.create_visualization(
                clean_mask.shape, object_data, micrometers_per_pixel, 
                main_folder, base_name, ".jpg") # Pasar extensión .jpg
        
        # Actualizar progreso (usar ruta original o la nueva jpg para thumbnail?)
        # Usemos la JPG para el progreso, ya que es la que se guarda
        self.progress_updated.emit(index, overlay_path)
    
    def extract_microns_per_pixel(self, image_path, filename):
        """Extrae la información de micrómetros por píxel de los metadatos."""
        import re
        import tifffile
        
        # Valor predeterminado
        micrometers_per_pixel = 1.0
        
        try:
            if filename.lower().endswith(('.tiff', '.tif')):
                with tifffile.TiffFile(image_path) as tif:
                    if 'ImageDescription' in tif.pages[0].tags:
                        desc = tif.pages[0].tags['ImageDescription'].value
                        
                        # Buscar MicronsPerPixel en JSON
                        if "MicronsPerPixel" in desc:
                            match = re.search(r'"MicronsPerPixel":\s*([\d.]+)', desc)
                            if match:
                                micrometers_per_pixel = float(match.group(1))
                        
                        # Buscar en el XML de OME
                        elif "<OME" in desc and "PhysicalSizeX" in desc:
                            match = re.search(r'PhysicalSizeX="([\d.]+)"', desc)
                            if match:
                                micrometers_per_pixel = float(match.group(1))
                    
                    # Buscar en tags estándar
                    if micrometers_per_pixel == 1.0 and 'XResolution' in tif.pages[0].tags:
                        x_resolution = tif.pages[0].tags['XResolution'].value
                        resolution_unit = tif.pages[0].tags.get('ResolutionUnit', None)
                        
                        if resolution_unit is not None:
                            if resolution_unit.value == 2:  # Pulgadas
                                micrometers_per_pixel = 25400.0 / float(x_resolution[0] / x_resolution[1])
                            elif resolution_unit.value == 3:  # Centímetros
                                micrometers_per_pixel = 10000.0 / float(x_resolution[0] / x_resolution[1])
        except Exception as e:
            print(f"Error obteniendo micrómetros por pixel: {e}")
        
        return micrometers_per_pixel
    
    def create_basic_mask(self, img, predictions, main_folder, base_name, save_extension):
        """Crea la máscara básica a partir de las predicciones y la guarda en JPG."""
        from landingai.visualize import overlay_colored_masks
        from PIL import Image

        mask_filename_jpg = f"{base_name}-mask{save_extension}" # Usar extensión pasada
        mask_path = os.path.join(main_folder, "masks", mask_filename_jpg)
        
        # Crear máscara básica
        black_img = Image.new("L", img.size, 0)

        if len(predictions) == 0:
            return black_img, mask_path
        
        # Crear el mapa de colores adecuado
        labels = [pred.label_name for pred in predictions]
        mask_alpha = {"mask_alpha": 1.0}
        color_map = {label: "white" for label in labels}
        options = {**color_map, **mask_alpha}  # Combinar color_map y mask_alpha
        
        mask_img = overlay_colored_masks(predictions, black_img, options)

        # Colorear máscara en blanco y negro
        mask_img = mask_img.convert("L") # Convertir a escala de grises para JPG
        # No convertir a modo "1", JPG no lo soporta bien. Se guarda como grayscale.
        # mask_img = mask_img.point(lambda p: 255 if p > 0 else 0, "1") # ELIMINADO

        # Guardar máscara en formato JPG
        mask_img.save(mask_path, format='JPEG', quality=90) # Calidad 90 (ajustable)
        
        # Retornar la máscara PIL en formato 'L' para el procesamiento posterior
        return mask_img, mask_path
    
    def process_mask(self, mask_img, main_folder, base_name, original_extension):
        """Procesa la máscara para separar objetos y eliminar objetos de borde."""
        import cv2
        import numpy as np
        
        # Convertir máscara PIL ('L') a formato numpy
        mask_np = np.array(mask_img) # Ya está en 'L' (escala de grises)
        # Aplicar umbral para asegurar que sea binaria internamente para el análisis
        _, binary_mask = cv2.threshold(mask_np, 1, 255, cv2.THRESH_BINARY) # Umbral bajo porque viene de JPG
        
        # Identificar objetos
        num_labels, labels = cv2.connectedComponents(binary_mask)
        
        # Eliminar objetos que tocan el borde
        height, width = binary_mask.shape
        border_mask = np.zeros((height, width), dtype=bool)
        border_mask[0, :] = True
        border_mask[height-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, width-1] = True
        
        # Identificar etiquetas que tocan el borde
        border_labels = set()
        for label in range(1, num_labels):
            if np.any(border_mask[labels == label]):
                border_labels.add(label)
        
        # Crear máscara limpia (sin objetos de borde)
        clean_mask = np.zeros_like(binary_mask)
        for label in range(1, num_labels):
            if label not in border_labels:
                clean_mask[labels == label] = 255
        
        return clean_mask
    
    def find_multiple_circles(self, mask):
        """Encuentra múltiples círculos máximos dentro de una máscara de objeto."""
        import cv2
        import numpy as np

        # Paraámetros configurables
        local_max_threshold = 0.5 # El umbral para detectar máximos locales. X veces el valor máximo de la transformada de distancia.
        max_allowed_superposition = 0.4 # La superposición máxima permitida está establecida en X veces la suma de los radios.

        # Transformada de distancia
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # Normalizar para mejor visualización
        if np.max(dist) > 0:
            dist_normalized = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
        else:
            return []  # Máscara vacía
        
        # Encontrar máximos locales
        kernel = np.ones((3, 3), np.uint8)
        local_max = cv2.dilate(dist, kernel)
        peaks = (dist == local_max) & (dist > local_max_threshold * np.max(dist))
        
        # Obtener coordenadas de máximos locales
        coordinates = list(zip(*np.where(peaks)))
        
        # Ordenar por valor de distancia (descendente)
        coordinates.sort(key=lambda p: dist[p[0], p[1]], reverse=True)
        
        # Almacenar círculos (center_y, center_x, radius)
        circles = []
        
        # Empezar con el círculo más grande
        if coordinates:
            y, x = coordinates[0]
            radius = dist[y, x]
            circles.append((y, x, radius))
            
            # Procesar los máximos locales restantes
            for y, x in coordinates[1:]:
                radius = dist[y, x]
                
                # Comprobar solapamiento con círculos existentes
                overlap_too_much = False
                for cy, cx, cr in circles:
                    # Calcular distancia euclidiana entre centros
                    center_dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                    
                    # Si los centros están más cerca del 30% de la suma de radios,
                    # considerar que se solapan demasiado
                    if center_dist < max_allowed_superposition * (radius + cr):
                        overlap_too_much = True
                        break
                
                # Añadir círculo si no se solapa demasiado
                if not overlap_too_much and radius > 3:  # Radio mínimo para evitar ruido
                    circles.append((y, x, radius))
        
        # Convertir a formato OpenCV (x, y, radius)
        return [(x, y, r) for y, x, r in circles]
    
    def analyze_objects(self, mask, micrometers_per_pixel, detailed_csv_path, filename):
        """Analiza los objetos en la máscara y calcula sus propiedades."""
        import cv2
        import numpy as np
        import csv
        from skimage.measure import regionprops
        
        # Obtener etiquetas finales
        final_num_labels, final_labels = cv2.connectedComponents(mask)
        
        # Listas para almacenar mediciones
        feret_diameters = []
        circle_diameters = []  # Nueva lista para diámetros de círculos máximos
        circularity_values = []
        object_data = []
        
        valid_object_count = 0
        
        # Procesar cada objeto
        for label_idx in range(1, final_num_labels):
            # Obtener máscara para este objeto
            object_mask = (final_labels == label_idx).astype(np.uint8)
            
            # Excluir objetos muy pequeños
            if np.sum(object_mask) < 30:
                continue
            
            valid_object_count += 1
            
            # Obtener propiedades
            props = regionprops(object_mask)
            if not props:
                continue
                
            props = props[0]
            
            # Calcular diámetro de Feret
            feret_diameter_px = props.feret_diameter_max
            feret_diameter_um = feret_diameter_px * micrometers_per_pixel
            feret_diameters.append(feret_diameter_um)
            
            # Intentar obtener los puntos exactos del diámetro de Feret
            feret_points = None
            if hasattr(props, 'feret_diameter_max_coordinates'):
                feret_points = props.feret_diameter_max_coordinates
                # Convertir coordenadas (row, col) a (x, y)
                feret_points = [(int(p[1]), int(p[0])) for p in feret_points]
            else:
                # Cálculo manual si no está disponible
                contours, _ = cv2.findContours(object_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    contour_points = contour.reshape(-1, 2)
                    
                    # Optimización para contornos grandes
                    if len(contour_points) > 100:
                        step = max(1, len(contour_points) // 100)
                        sparse_points = contour_points[::step]
                    else:
                        sparse_points = contour_points
                    
                    # Encontrar la distancia máxima
                    max_dist = 0
                    for i in range(len(sparse_points)):
                        for j in range(i+1, len(sparse_points)):
                            dist = np.sqrt(np.sum((sparse_points[i] - sparse_points[j])**2))
                            if dist > max_dist:
                                max_dist = dist
                                feret_points = (
                                    (int(sparse_points[i][0]), int(sparse_points[i][1])),
                                    (int(sparse_points[j][0]), int(sparse_points[j][1]))
                                )
            
            # Obtener contorno
            contours, _ = cv2.findContours(object_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Encontrar círculo máximo inscrito
            dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
            
            # Calcular circularidad
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                circularity_values.append(circularity)
                
                # Encuentra múltiples círculos dentro del objeto
                detected_circles = self.find_multiple_circles(object_mask)
                
                # Calcular diámetro del círculo máximo en micrómetros
                circle_diameter_um = max_val * 2 * micrometers_per_pixel
                circle_diameters.append(circle_diameter_um)
                
                # Guardar datos del objeto
                object_data.append({
                    'contour': contour,
                    'feret_diameter': feret_diameter_um,
                    'circle_diameter': circle_diameter_um,
                    'feret_points': feret_points,
                    'circularity': circularity,
                    'max_circle_center': max_loc,
                    'max_circle_radius': max_val,
                    'object_number': valid_object_count,
                    'detected_circles': detected_circles
                })
                
                # Escribir en CSV detallado
                with open(detailed_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        filename,
                        valid_object_count,
                        f"{feret_diameter_um:.2f}",
                        f"{circle_diameter_um:.2f}",
                        f"{circularity:.4f}"
                    ])
        
        return object_data, feret_diameters, circle_diameters, circularity_values
    
    def write_summary_csv(self, summary_csv_path, filename, object_data, feret_diameters, circle_diameters, circularity_values):
        """Escribe el resumen de mediciones en el archivo CSV."""
        import csv
        import numpy as np
        
        # Calcular promedios
        avg_feret = np.mean(feret_diameters) if feret_diameters else 0
        avg_circle = np.mean(circle_diameters) if circle_diameters else 0  # Promedio del diámetro del círculo
        avg_circularity = np.mean(circularity_values) if circularity_values else 0
        
        # Escribir en CSV de resumen
        with open(summary_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                filename,
                len(object_data),
                f"{avg_feret:.2f}",
                f"{avg_circle:.2f}",  # Añadir promedio del diámetro del círculo
                f"{avg_circularity:.4f}"
            ])
    
    def create_visualization(self, shape, object_data, micrometers_per_pixel, main_folder, base_name, save_extension):
        """Crea una visualización con anotaciones de los objetos y la guarda en JPG."""
        import cv2
        import numpy as np
        
        height, width = shape
        
        # Crear máscara para visualización
        calc_mask = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Dibujar objetos
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for obj_data in object_data:
            contour = obj_data['contour']
            feret_diam = obj_data['feret_diameter']
            feret_points = obj_data['feret_points']
            circularity = obj_data['circularity']
            circle_center = obj_data['max_circle_center']
            circle_radius = obj_data['max_circle_radius']
            object_number = obj_data['object_number']
            detected_circles = obj_data.get('detected_circles', [])
            
            # Dibujar objeto
            cv2.drawContours(calc_mask, [contour], -1, (255, 255, 255), -1)
            cv2.drawContours(calc_mask, [contour], -1, (0, 255, 0), 2)
            
            # Calcular centroide
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = circle_center
            
            # Dibujar múltiples círculos en rosa
            for circle_x, circle_y, radius in detected_circles:
                cv2.circle(calc_mask, (circle_x, circle_y), int(radius), (180, 105, 255), 2)
            
            # Dibujar diámetro de Feret
            if feret_points:
                cv2.line(calc_mask, feret_points[0], feret_points[1], (0, 0, 255), 2)
                # Añadir pequeños círculos en los extremos
                cv2.circle(calc_mask, feret_points[0], 3, (0, 0, 255), -1)
                cv2.circle(calc_mask, feret_points[1], 3, (0, 0, 255), -1)
            
            # Dibujar círculo máximo inscrito
            cv2.circle(calc_mask, circle_center, int(circle_radius), (255, 0, 0), 2)
            
            # Calcular ancho para posicionar texto
            rect = cv2.minAreaRect(contour)
            width_rect = max(rect[1][0], rect[1][1])
            
            # Añadir textos
            text_x = int(cX + width_rect/2)
            cv2.putText(calc_mask, f"#{object_number}", (text_x, cY - 35), font, 0.7, (255, 255, 0), 2)
            cv2.putText(calc_mask, f"Feret: {feret_diam:.2f} um", (text_x, cY - 15), font, 0.6, (255, 255, 255), 2)
            cv2.putText(calc_mask, f"Circ: {circularity:.4f}", (text_x, cY + 15), font, 0.6, (255, 255, 255), 2)
            
            # Añadir número de círculos detectados
            if len(detected_circles) > 1:
                cv2.putText(calc_mask, f"Círculos: {len(detected_circles)}", (text_x, cY + 45), 
                          font, 0.6, (180, 105, 255), 2)
        
        # Añadir información de escala
        cv2.putText(calc_mask, f"Escala: {micrometers_per_pixel:.2f} um/px", (10, height - 20), 
                   font, 0.7, (255, 255, 255), 2)
        
        # Guardar máscara con cálculos en formato JPG
        calc_mask_filename_jpg = f"{base_name}-calc{save_extension}" # Usar extensión pasada
        calc_mask_path = os.path.join(main_folder, "masks_calculations", calc_mask_filename_jpg)
        # Especificar parámetros de calidad para JPG
        cv2.imwrite(calc_mask_path, calc_mask, [cv2.IMWRITE_JPEG_QUALITY, 90]) # Calidad 90 (ajustable)

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_folder = None
        self.api_key = ""
        self.project_id = ""
        self.current_image_path = None
        self.current_original_path = None  # Ruta de la imagen original
        self.current_processed_path = None  # Ruta de la imagen procesada
        self.current_mask_path = None
        self.current_mask_calc_path = None
        
        # Detectar si estamos en modo oscuro o claro
        app = QApplication.instance()
        self.is_dark_mode = self.is_system_dark_mode()
        
        # Cargar fuentes y estilos
        self.setup_ui_resources()
        self.load_settings()
        self.initUI()
        
        # Configurar atajos de teclado
        self.setup_shortcuts()
    
    def is_system_dark_mode(self):
        app = QApplication.instance()
        # Obtener la paleta del sistema
        palette = app.palette()
        # Comprobar si el color de fondo del sistema es oscuro
        bg_color = palette.color(QPalette.ColorRole.Window)
        # Si la luminosidad es baja, estamos en modo oscuro
        return bg_color.lightness() < 128
    
    def setup_ui_resources(self):
        # Aplicar estilos según el modo
        self.setStyleSheet(get_app_styles(self.is_dark_mode))
        
        # Configurar la fuente predeterminada
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
    
    def setup_shortcuts(self):
        # Atajo para procesar imágenes (Ctrl+P)
        shortcut_process = QShortcut(QKeySequence("Ctrl+P"), self)
        shortcut_process.activated.connect(self.process_images)
        
        # Atajo para seleccionar carpeta (Ctrl+O)
        shortcut_folder = QShortcut(QKeySequence("Ctrl+O"), self)
        shortcut_folder.activated.connect(self.select_folder)
        
        # Atajo para alternar entre pestañas (Ctrl+Tab)
        shortcut_tab = QShortcut(QKeySequence("Ctrl+I"), self)
        shortcut_tab.activated.connect(self.toggle_tab)
    
    def toggle_tab(self):
        current_index = self.image_type_selector.currentIndex()
        new_index = 1 - current_index  # Alternar entre 0 y 1
        self.image_type_selector.setCurrentIndex(new_index)
    
    def initUI(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        
        left_layout = QVBoxLayout()
        left_layout.setSpacing(15)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        # Título de la aplicación
        app_title = QLabel("PROCESADOR DE IMÁGENES")
        app_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = app_title.font()
        font.setPointSize(14)
        font.setBold(True)
        app_title.setFont(font)
        left_layout.addWidget(app_title)
        
        # Sección de configuración
        config_title = QLabel("Configuración")
        config_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        font = config_title.font()
        font.setBold(True)
        config_title.setFont(font)
        left_layout.addWidget(config_title)
        
        # Botón de configuración
        self.btn_config = QPushButton("Configurar API Key y Endpoint ID")
        self.btn_config.setIcon(QIcon.fromTheme("configure"))
        self.btn_config.clicked.connect(self.show_config_dialog)
        left_layout.addWidget(self.btn_config)
        
        # Añadir una etiqueta para mostrar estado de configuración (sin detalles de API key)
        self.config_status_label = QLabel("Estado: No configurado")
        self.config_status_label.setWordWrap(True)
        left_layout.addWidget(self.config_status_label)
        self.update_config_status()
        
        self.btn_select_folder = QPushButton("Seleccionar Carpeta de Imágenes")
        self.btn_select_folder.setIcon(QIcon.fromTheme("folder-open"))
        self.btn_select_folder.clicked.connect(self.select_folder)
        left_layout.addWidget(self.btn_select_folder)
        
        self.folder_info_label = QLabel("No hay carpeta seleccionada")
        self.folder_info_label.setWordWrap(True)
        left_layout.addWidget(self.folder_info_label)
        
        # Sección de procesamiento
        process_title = QLabel("Procesamiento")
        process_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        font = process_title.font()
        font.setBold(True)
        process_title.setFont(font)
        left_layout.addWidget(process_title)
        
        self.btn_process = QPushButton("Procesar Imágenes")
        self.btn_process.setIcon(QIcon.fromTheme("system-run"))
        self.btn_process.clicked.connect(self.process_images)
        self.btn_process.setEnabled(False)
        left_layout.addWidget(self.btn_process)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v/%m imágenes (%p%)")
        left_layout.addWidget(self.progress_bar)
        
        # Sección de imágenes procesadas
        images_title = QLabel("Imágenes Procesadas")
        images_title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        font = images_title.font()
        font.setBold(True)
        images_title.setFont(font)
        left_layout.addWidget(images_title)
        
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QPixmap(100, 100).size())
        self.image_list.setSpacing(10)
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.itemClicked.connect(self.show_large_image)
        self.image_list.currentItemChanged.connect(self.show_large_image)
        left_layout.addWidget(self.image_list)
        
        # Crear el contenedor de pestañas para la visualización de imágenes
        right_container = QWidget()
        right_container_layout = QVBoxLayout(right_container)
        right_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Añadir TabWidget para selección de tipo de imagen
        self.image_type_selector = QTabWidget()
        self.image_type_selector.setDocumentMode(True)  # Hace que las pestañas se vean más modernas
        
        # Crear páginas para cada tipo de imagen
        self.processed_tab = QWidget()
        self.original_tab = QWidget()
        processed_layout = QVBoxLayout(self.processed_tab)
        processed_layout.setContentsMargins(0, 0, 0, 0)
        original_layout = QVBoxLayout(self.original_tab)
        original_layout.setContentsMargins(0, 0, 0, 0)
        
        # Configurar los labels de imagen grande para cada pestaña
        self.processed_image_label = QLabel("Seleccione una imagen para verla en grande")
        self.original_image_label = QLabel("Seleccione una imagen para verla en grande")
        self.active_label = self.original_image_label
        
        for label in [self.processed_image_label, self.original_image_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(400, 400)
            label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            # Añadir estilo para el borde
            if self.is_dark_mode:
                label.setStyleSheet("border: 1px solid #444444; border-radius: 8px; background-color: #2D2D30;")
            else:
                label.setStyleSheet("border: 1px solid #DDDDDD; border-radius: 8px; background-color: #FFFFFF;")
        
        # Añadir los labels a sus respectivas pestañas
        processed_layout.addWidget(self.processed_image_label)
        original_layout.addWidget(self.original_image_label)
        
        # Añadir nuevas pestañas para máscara y máscara con datos
        self.mask_tab = QWidget()
        self.mask_calc_tab = QWidget()
        mask_layout = QVBoxLayout(self.mask_tab)
        mask_calc_layout = QVBoxLayout(self.mask_calc_tab)
        mask_layout.setContentsMargins(0, 0, 0, 0)
        mask_calc_layout.setContentsMargins(0, 0, 0, 0)

        # Añadir las pestañas al selector en el orden correcto
        self.image_type_selector.addTab(self.processed_tab, "Imagen Procesada")
        self.image_type_selector.addTab(self.original_tab, "Imagen Original")
        self.image_type_selector.addTab(self.mask_tab, "Máscara")
        self.image_type_selector.addTab(self.mask_calc_tab, "Máscara con Datos")
        
        # Asegurar que la primera pestaña (Imagen Procesada) está seleccionada
        self.image_type_selector.setCurrentIndex(0)
        
        # Configurar los labels de imagen para las nuevas pestañas
        self.mask_image_label = QLabel("Seleccione una imagen para ver su máscara")
        self.mask_calc_image_label = QLabel("Seleccione una imagen para ver su máscara con datos")
        
        for label in [self.mask_image_label, self.mask_calc_image_label]:
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumSize(400, 400)
            label.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding
            )
            # Añadir estilo para el borde
            if self.is_dark_mode:
                label.setStyleSheet("border: 1px solid #444444; border-radius: 8px; background-color: #2D2D30;")
            else:
                label.setStyleSheet("border: 1px solid #DDDDDD; border-radius: 8px; background-color: #FFFFFF;")
        
        # Añadir los labels a sus respectivas pestañas
        mask_layout.addWidget(self.mask_image_label)
        mask_calc_layout.addWidget(self.mask_calc_image_label)
        
        self.image_type_selector.currentChanged.connect(self.on_tab_changed)
        
        # Añadir el selector de pestañas y el label de nombre al layout derecho
        right_container_layout.addWidget(self.image_type_selector)
        
        # Añadir label para el nombre de la imagen
        self.image_name_label = QLabel()
        self.image_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_name_label.setWordWrap(True)
        # Estilizar el label de nombre
        font = self.image_name_label.font()
        font.setPointSize(11)
        self.image_name_label.setFont(font)
        right_container_layout.addWidget(self.image_name_label)
        
        # Configurar el splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        left_widget.setMinimumWidth(310)
        left_widget.setMaximumWidth(310)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_container)
        splitter.setSizes([310, 500])
        main_layout.addWidget(splitter)
        
        self.setWindowTitle("Laboratorio de Procesamiento de Imágenes")
        self.setGeometry(100, 100, 1200, 700)  # Ventana más grande por defecto
        
        # Habilitar drag & drop en la lista de miniaturas
        self.image_list.setDragEnabled(True)
        self.image_list.setDragDropMode(QListWidget.DragDropMode.DragOnly)
    
    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta")
        if folder:
            self.image_folder = folder
            # Get original images (TIFF/TIF)
            original_images = [f for f in os.listdir(folder) if f.lower().endswith((".tiff", ".tif"))]
            total_original_images = len(original_images)
            
            folder_name = os.path.basename(folder)
            self.folder_info_label.setText(f"Carpeta: {folder_name}\nTotal imágenes originales: {total_original_images}")
            
            # Set progress bar maximum to the number of original images
            self.progress_bar.setMaximum(total_original_images if total_original_images > 0 else 1) # Avoid max=0
            self.progress_bar.setValue(0)
            self.image_list.clear() # Clear previous thumbnails
            
            # Find existing processed folders
            processed_folders = []
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isdir(item_path) and item.startswith("processed_images_") and len(item.split('_')) == 3:
                    try:
                        # Extract timestamp for sorting
                        timestamp_str = item.split('_', 2)[-1]
                        # Attempt to parse the timestamp to ensure valid format
                        datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S") 
                        processed_folders.append((timestamp_str, item_path))
                    except ValueError:
                        # Ignore folders that don't match the expected timestamp format
                        continue
            
            if processed_folders:
                # Sort by timestamp (most recent first)
                processed_folders.sort(key=lambda x: x[0], reverse=True)
                
                most_recent_folder_path = processed_folders[0][1]
                most_recent_folder_name = os.path.basename(most_recent_folder_path)
                overlay_images_path = os.path.join(most_recent_folder_path, "overlayed_images")
                
                # Check if the overlayed_images subfolder exists
                if os.path.isdir(overlay_images_path):
                    # Load existing processed images (JPGs)
                    processed_jpgs = [f for f in os.listdir(overlay_images_path) if f.lower().endswith(".jpg")]
                    
                    num_processed = len(processed_jpgs)
                    if num_processed > 0:
                        for image_name in processed_jpgs:
                            image_path = os.path.join(overlay_images_path, image_name)
                            self.add_thumbnail(image_path)
                        
                        # Update progress bar based on loaded processed images relative to originals
                        # If #processed >= #originals, assume all are done
                        progress_value = min(num_processed, total_original_images)
                        self.progress_bar.setValue(progress_value)
                        
                        # Update the info label
                        self.folder_info_label.setText(
                            f"Carpeta: {folder_name}\n"
                            f"Total imágenes originales: {total_original_images}\n"
                            f"Se cargaron {num_processed} resultados previos de: {most_recent_folder_name}"
                        )
                    else:
                        self.folder_info_label.setText(
                             f"Carpeta: {folder_name}\n"
                             f"Total imágenes originales: {total_original_images}\n"
                             f"Carpeta procesada encontrada ({most_recent_folder_name}), pero sin imágenes JPG."
                         )
                else:
                    self.folder_info_label.setText(
                         f"Carpeta: {folder_name}\n"
                         f"Total imágenes originales: {total_original_images}\n"
                         f"Carpeta procesada encontrada ({most_recent_folder_name}), pero falta subcarpeta 'overlayed_images'."
                     )

            # Enable process button only if there are original images to process
            self.btn_process.setEnabled(total_original_images > 0)
            if total_original_images == 0 and not processed_folders:
                self.folder_info_label.setText(f"Carpeta: {folder_name}\nNo hay imágenes originales (.tiff/.tif) para procesar.")
    
    def process_images(self):
        if hasattr(self, 'thread') and isinstance(self.thread, ImageProcessorThread) and self.thread.isRunning():
            # Si el proceso está en marcha, lo detenemos
            self.thread.stop()
            self.thread.wait()
            self.btn_process.setText("Procesar Imágenes")
            return
        
        if not self.image_folder:
            return
        
        if not self.api_key or not self.project_id:
            QMessageBox.warning(self, "Configuración incompleta", 
                               "Por favor, configura la API Key y el Endpoint ID antes de procesar imágenes.")
            self.show_config_dialog()
            return
        
        self.image_list.clear()
        self.progress_bar.setValue(0)
        
        self.thread = ImageProcessorThread(self.image_folder, self.api_key, self.project_id)
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.processing_done.connect(self.processing_finished)
        self.thread.error_occurred.connect(self.handle_processing_error)
        self.thread.start()
        
        self.btn_process.setText("Detener proceso")
    
    def update_progress(self, current_image, image_path):
        self.progress_bar.setValue(current_image)
        
        # Determinar si es un path procesado
        if "overlayed_images" in image_path:
            self.add_thumbnail(image_path)
    
    def processing_finished(self):
        self.progress_bar.setValue(100)
        self.btn_process.setText("Procesar Imágenes")
    
    def add_thumbnail(self, image_path):
        pixmap = QPixmap(image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
        icon = QIcon(pixmap)
        item = QListWidgetItem()
        item.setIcon(icon)
        item.setData(Qt.ItemDataRole.UserRole, image_path)
        item.setToolTip(os.path.basename(image_path))  # Añadir tooltip con el nombre
        self.image_list.addItem(item)
    
    def show_large_image(self, item):
        if item:
            # Obtener la ruta de la imagen procesada (overlay) que ahora es JPG
            processed_path = item.data(Qt.ItemDataRole.UserRole)
            self.current_processed_path = processed_path
            
            # Deducir las rutas de las demás imágenes
            overlay_filename = os.path.basename(processed_path) # e.g., imagen.jpg
            base_name = os.path.splitext(overlay_filename)[0] # e.g., imagen
            folder_path = os.path.dirname(processed_path) # .../processed_images_XXX/overlayed_images
            parent_folder = os.path.dirname(folder_path) # .../processed_images_XXX
            original_images_folder = os.path.dirname(parent_folder) # Carpeta original de imágenes
            
            # Buscar la imagen original (probablemente TIFF)
            # Necesitamos encontrar el archivo original que generó este JPG
            original_tiff_path = os.path.join(original_images_folder, f"{base_name}.tiff")
            original_tif_path = os.path.join(original_images_folder, f"{base_name}.tif")
            if os.path.exists(original_tiff_path):
                self.current_original_path = original_tiff_path
            elif os.path.exists(original_tif_path):
                 self.current_original_path = original_tif_path
            else:
                 # Si no se encuentra TIFF/TIF, buscar JPG/PNG original como respaldo
                 original_jpg_path = os.path.join(original_images_folder, f"{base_name}.jpg")
                 original_png_path = os.path.join(original_images_folder, f"{base_name}.png")
                 if os.path.exists(original_jpg_path):
                     self.current_original_path = original_jpg_path
                 elif os.path.exists(original_png_path):
                     self.current_original_path = original_png_path
                 else:
                     self.current_original_path = None # No se encontró original
            
            # Obtener la ruta de la máscara (ahora JPG)
            mask_folder = os.path.join(parent_folder, "masks")
            mask_filename = f"{base_name}-mask.jpg" # Extensión .jpg
            mask_path = os.path.join(mask_folder, mask_filename)
            self.current_mask_path = mask_path if os.path.exists(mask_path) else None
            
            # Obtener la ruta de la máscara con cálculos (ahora JPG)
            mask_calc_folder = os.path.join(parent_folder, "masks_calculations")
            mask_calc_filename = f"{base_name}-calc.jpg" # Extensión .jpg
            mask_calc_path = os.path.join(mask_calc_folder, mask_calc_filename)
            self.current_mask_calc_path = mask_calc_path if os.path.exists(mask_calc_path) else None
            
            # Actualizar el nombre mostrado (usar el base_name)
            self.image_name_label.setText(base_name) # Mostrar nombre base sin extensión
            
            # Actualizar la imagen según la pestaña actual
            self.on_tab_changed()
    
    def on_tab_changed(self):
        index = self.image_type_selector.currentIndex()
        
        if index == 0:  # Pestaña "Imagen Procesada"
            self.current_image_path = self.current_processed_path
            self.active_label = self.processed_image_label
        elif index == 1:  # Pestaña "Imagen Original"
            self.current_image_path = self.current_original_path
            self.active_label = self.original_image_label
        elif index == 2:  # Pestaña "Máscara"
            self.current_image_path = self.current_mask_path
            self.active_label = self.mask_image_label
        elif index == 3:  # Pestaña "Máscara con Datos"
            self.current_image_path = self.current_mask_calc_path
            self.active_label = self.mask_calc_image_label
        
        self.update_large_image()
    
    def update_large_image(self):
        if not self.current_image_path or not os.path.exists(self.current_image_path):
            self.active_label.setText("Imagen no disponible")
            return
            
        original_pixmap = QPixmap(self.current_image_path)
        label_size = self.active_label.size()
        
        scaled_size = original_pixmap.size()
        scaled_size.scale(label_size.width(), label_size.height(), Qt.AspectRatioMode.KeepAspectRatio)
        
        final_width = min(scaled_size.width(), original_pixmap.width())
        final_height = min(scaled_size.height(), original_pixmap.height())
        
        pixmap = original_pixmap.scaled(
            final_width,
            final_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.active_label.setPixmap(pixmap)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_large_image()
    
    def save_settings(self):
        # Actualizar el nombre de la clave: "project_id" a "endpoint_id"
        settings = {"api_key": self.api_key, "endpoint_id": self.project_id}
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    
    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                self.api_key = settings.get("api_key", "")
                # Manejar tanto el nuevo nombre (endpoint_id) como el antiguo (project_id)
                self.project_id = settings.get("endpoint_id", settings.get("project_id", ""))
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.current_image_path:
            # Obtener el label activo según la pestaña
            active_label = self.processed_image_label if self.image_type_selector.currentIndex() == 0 else self.original_image_label
            
            # Comprobar si el clic fue dentro del QLabel de la imagen grande
            if active_label.geometry().contains(event.pos()):
                # Crear el objeto de arrastre
                drag = QDrag(self)
                mime_data = QMimeData()
                
                # Añadir la ruta del archivo actual (original o procesado según selección)
                urls = [QUrl.fromLocalFile(self.current_image_path)]
                mime_data.setUrls(urls)
                
                # Crear una miniatura para mostrar durante el arrastre
                pixmap = QPixmap(self.current_image_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                drag.setPixmap(pixmap)
                
                drag.setMimeData(mime_data)
                drag.exec(Qt.DropAction.CopyAction)
                
        super().mousePressEvent(event)

    def handle_processing_error(self, error_message):
        # Mostrar mensaje de error
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Icon.Critical)
        error_dialog.setWindowTitle("Error de Procesamiento")
        error_dialog.setText(error_message)
        error_dialog.setStandardButtons(QMessageBox.StandardButton.Retry | QMessageBox.StandardButton.Cancel)
        
        # Restablecer UI
        self.btn_process.setText("Procesar Imágenes")
        
        # Mostrar diálogo y manejar respuesta
        result = error_dialog.exec()
        
        if result == QMessageBox.StandardButton.Retry:
            # Reintentar procesamiento
            self.process_images()

    def show_config_dialog(self):
        """Muestra el diálogo de configuración como una ventana independiente"""
        config_dialog = ConfigDialog(self, self.api_key, self.project_id)
        
        # Ejecutar el diálogo y capturar el resultado
        if config_dialog.exec() == QDialog.DialogCode.Accepted:
            self.api_key = config_dialog.api_key
            self.project_id = config_dialog.endpoint_id
            self.save_settings()
            self.update_config_status()
    
    def update_config_status(self):
        """Actualiza la etiqueta de estado de configuración sin mostrar datos sensibles"""
        if self.api_key and self.project_id:
            self.config_status_label.setText("Estado: Configurado ✓")
            self.config_status_label.setStyleSheet("color: green;")
        else:
            self.config_status_label.setText("Estado: No configurado ✗")
            self.config_status_label.setStyleSheet("color: red;")

class ConfigDialog(QDialog):
    def __init__(self, parent=None, api_key="", endpoint_id=""):
        super().__init__(parent)
        self.setWindowTitle("Configuración")
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.result = False
        
        # Configurar la ventana para que sea independiente y tenga controles de ventana
        self.setWindowFlag(Qt.WindowType.Window)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint)
        self.setWindowFlag(Qt.WindowType.WindowMinimizeButtonHint)
        
        self.initUI()
    
    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # Título
        title = QLabel("Configuración de Landing AI")
        title_font = title.font()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Campo de API Key
        api_key_label = QLabel("API Key:")
        self.api_key_input = QLineEdit(self.api_key)
        self.api_key_input.setPlaceholderText("Introduce tu API Key de Landing AI")
        layout.addWidget(api_key_label)
        layout.addWidget(self.api_key_input)
        
        # Campo de Endpoint ID
        endpoint_id_label = QLabel("Endpoint ID:")
        self.endpoint_id_input = QLineEdit(self.endpoint_id)
        self.endpoint_id_input.setPlaceholderText("Introduce el ID del endpoint")
        layout.addWidget(endpoint_id_label)
        layout.addWidget(self.endpoint_id_input)
        
        # Botones
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Guardar")
        self.cancel_button = QPushButton("Cancelar")
        
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
        
        self.setMinimumWidth(350)
        self.setFixedHeight(layout.sizeHint().height())
    
    def accept(self):
        self.api_key = self.api_key_input.text().strip()
        self.endpoint_id = self.endpoint_id_input.text().strip()
        self.result = True
        super().accept()  # Usar el método estándar de QDialog
    
    def reject(self):
        self.result = False
        super().reject()  # Usar el método estándar de QDialog

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Configurar el estilo de la aplicación
    app.setStyle(QStyleFactory.create("Fusion"))
    
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
