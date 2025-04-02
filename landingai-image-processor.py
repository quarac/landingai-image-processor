import sys
import os
import torch
import json
import numpy as np
from PIL import Image, ImageOps
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem, QHBoxLayout, QLineEdit, QSplitter, QProgressBar, QSizePolicy, QTabWidget, QStyleFactory, QMessageBox)
from PyQt6.QtGui import QPixmap, QIcon, QDrag, QFontDatabase, QFont, QPalette, QColor, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QMimeData
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
    
    def stop(self):
        self.should_stop = True
    
    def run(self):
        from datetime import datetime
        import re
        import cv2
        import numpy as np
        import csv
        from skimage.measure import regionprops
        
        try:
            # Create main folder with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            main_folder = os.path.join(self.image_folder, f"processed_images_{timestamp}")
            
            # Create directory structure
            os.makedirs(main_folder, exist_ok=True)
            overlay_folder = os.path.join(main_folder, "overlayed_images")
            masks_folder = os.path.join(main_folder, "masks")
            masks_calc_folder = os.path.join(main_folder, "masks_calculations")
            
            os.makedirs(overlay_folder, exist_ok=True)
            os.makedirs(masks_folder, exist_ok=True)
            os.makedirs(masks_calc_folder, exist_ok=True)
            
            # Create CSV files
            detailed_csv_path = os.path.join(main_folder, "detailed_results.csv")
            summary_csv_path = os.path.join(main_folder, "summary_results.csv")
            
            # Initialize CSV files
            with open(detailed_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Imagen', 'Número de objeto', 'Diámetro máximo de Feret (um)', 'Circularidad'])
            
            with open(summary_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Imagen', 'Número de objetos', 'Promedio diámetro de Feret (um)', 'Promedio circularidad'])
            
            # Process images
            images = [f for f in os.listdir(self.image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif"))]
            total_images = len(images)
            
            try:
                predictor = Predictor(endpoint_id=self.project_id, api_key=self.api_key)
            except ConnectionError as e:
                self.error_occurred.emit("Error de conexión con LandingAI. Comprueba tu conexión a internet y que la API Key sea correcta.")
                return
            except Exception as e:
                self.error_occurred.emit(f"Error al inicializar el predictor: {str(e)}")
                return
            
            for i, filename in enumerate(images, start=1):
                if self.should_stop:
                    break
                    
                try:
                    image_path = os.path.join(self.image_folder, filename)
                    
                    # Get microns per pixel
                    micrometers_per_pixel = 1.0  # Default value
                    try:
                        if filename.lower().endswith(('.tiff', '.tif')):
                            with tifffile.TiffFile(image_path) as tif:
                                if 'ImageDescription' in tif.pages[0].tags:
                                    desc = tif.pages[0].tags['ImageDescription'].value
                                    # Look for MicronsPerPixel in JSON within XML comments
                                    if "MicronsPerPixel" in desc:
                                        match = re.search(r'"MicronsPerPixel":\s*([\d.]+)', desc)
                                        if match:
                                            micrometers_per_pixel = float(match.group(1))
                    except Exception as e:
                        print(f"Error getting microns per pixel: {e}")
                    
                    # Load and process image
                    img = Image.open(image_path).convert("RGB")
                    predictions = predictor.predict(img)
                    
                    # Save overlayed image
                    overlayed_img = overlay_predictions(predictions, img)
                    overlay_save_path = os.path.join(overlay_folder, filename)
                    overlayed_img.save(overlay_save_path)
                    
                    # Create basic mask
                    black_img = Image.new("L", img.size, 0)
                    labels = [pred.label_name for pred in predictions]
                    mask_alpha = {"mask_alpha": 1.0}
                    color_map = {label: "white" for label in labels}
                    options = {**color_map, **mask_alpha}
                    mask_img = overlay_colored_masks(predictions, black_img, options)
                    
                    # Save basic mask
                    base_name = os.path.splitext(filename)[0]
                    original_extension = os.path.splitext(filename)[1]
                    mask_save_path = os.path.join(masks_folder, f"{base_name}-mask{original_extension}")
                    mask_img.save(mask_save_path)
                    
                    # Process mask
                    mask_np = np.array(mask_img.convert("L"))
                    _, binary_mask = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
                    
                    # Get connected components
                    num_labels, labels = cv2.connectedComponents(binary_mask)
                    
                    # Create border mask
                    height, width = binary_mask.shape
                    border_mask = np.zeros((height, width), dtype=bool)
                    border_mask[0, :] = True
                    border_mask[height-1, :] = True
                    border_mask[:, 0] = True
                    border_mask[:, width-1] = True
                    
                    # Remove border objects
                    border_labels = set()
                    for label in range(1, num_labels):
                        if np.any(border_mask[labels == label]):
                            border_labels.add(label)
                    
                    # Create clean mask
                    clean_mask = np.zeros_like(binary_mask)
                    for label in range(1, num_labels):
                        if label not in border_labels:
                            clean_mask[labels == label] = 255
                    
                    # Apply watershed
                    if np.any(clean_mask):
                        dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
                        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
                        
                        threshold_value = 0.5 * dist_transform.max()
                        if threshold_value == 0:
                            threshold_value = 0.1
                        
                        _, sure_fg = cv2.threshold(dist_transform, threshold_value, 255, 0)
                        sure_fg = sure_fg.astype(np.uint8)
                        
                        sure_bg = cv2.dilate(clean_mask, None, iterations=2)
                        unknown = cv2.subtract(sure_bg, sure_fg)
                        
                        ret, markers = cv2.connectedComponents(sure_fg)
                        markers = markers + 1
                        markers[unknown == 255] = 0
                        
                        watershed_input = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
                        markers = cv2.watershed(watershed_input, markers)
                        
                        watershed_mask = np.zeros_like(clean_mask)
                        watershed_mask[markers > 1] = 255
                    else:
                        watershed_mask = clean_mask
                    
                    # Get final labels
                    final_num_labels, final_labels = cv2.connectedComponents(watershed_mask)
                    
                    # Measure objects
                    feret_diameters = []
                    circularity_values = []
                    object_data = []
                    
                    valid_object_count = 0
                    
                    for label_idx in range(1, final_num_labels):
                        # Get object mask
                        object_mask = (final_labels == label_idx).astype(np.uint8)
                        
                        # Skip small objects
                        if np.sum(object_mask) < 30:
                            continue
                        
                        valid_object_count += 1
                        
                        # Get properties
                        props = regionprops(object_mask)
                        if not props:
                            continue
                            
                        props = props[0]
                        
                        # Calculate Feret diameter
                        feret_diameter_px = props.feret_diameter_max
                        feret_diameter_um = feret_diameter_px * micrometers_per_pixel
                        feret_diameters.append(feret_diameter_um)
                        
                        # Get contour
                        contours, _ = cv2.findContours(object_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue
                            
                        contour = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(contour)
                        perimeter = cv2.arcLength(contour, True)
                        
                        # Find max inscribed circle
                        dist_transform = cv2.distanceTransform(object_mask, cv2.DIST_L2, 5)
                        _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
                        
                        # Calculate circularity
                        if perimeter > 0:
                            circularity = 4 * np.pi * (area / (perimeter * perimeter))
                            circularity_values.append(circularity)
                            
                            # Store object data
                            object_data.append({
                                'contour': contour,
                                'feret_diameter': feret_diameter_um,
                                'circularity': circularity,
                                'max_circle_center': max_loc,
                                'max_circle_radius': max_val,
                                'object_number': valid_object_count
                            })
                            
                            # Write to detailed CSV
                            with open(detailed_csv_path, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    filename,
                                    valid_object_count,
                                    f"{feret_diameter_um:.2f}",
                                    f"{circularity:.4f}"
                                ])
                    
                    # Calculate averages
                    avg_feret = np.mean(feret_diameters) if feret_diameters else 0
                    avg_circularity = np.mean(circularity_values) if circularity_values else 0
                    
                    # Write to summary CSV
                    with open(summary_csv_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            filename,
                            valid_object_count,
                            f"{avg_feret:.2f}",
                            f"{avg_circularity:.4f}"
                        ])
                    
                    # Create calculation mask
                    calc_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    
                    # Draw objects
                    for obj_data in object_data:
                        contour = obj_data['contour']
                        feret_diam = obj_data['feret_diameter']
                        circularity = obj_data['circularity']
                        circle_center = obj_data['max_circle_center']
                        circle_radius = obj_data['max_circle_radius']
                        object_number = obj_data['object_number']
                        
                        # Draw object
                        cv2.drawContours(calc_mask, [contour], -1, (255, 255, 255), -1)
                        cv2.drawContours(calc_mask, [contour], -1, (0, 255, 0), 2)
                        
                        # Get centroid
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = circle_center
                        
                        # Draw Feret line using min area rect
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        width_rect = max(rect[1][0], rect[1][1])
                        
                        # Find maximum distance points
                        max_dist = 0
                        max_pts = None
                        for ii in range(4):
                            for jj in range(ii+1, 4):
                                dist = np.sqrt((box[ii][0] - box[jj][0])**2 + (box[ii][1] - box[jj][1])**2)
                                if dist > max_dist:
                                    max_dist = dist
                                    max_pts = (box[ii], box[jj])
                        
                        if max_pts:
                            cv2.line(calc_mask, tuple(max_pts[0]), tuple(max_pts[1]), (0, 0, 255), 2)
                        
                        # Draw max inscribed circle
                        cv2.circle(calc_mask, circle_center, int(circle_radius), (255, 0, 0), 2)
                        
                        # Add text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_x = int(cX + width_rect/2)  # Position text to the right
                        cv2.putText(calc_mask, f"#{object_number}", (text_x, cY - 35), font, 0.7, (255, 255, 0), 2)
                        cv2.putText(calc_mask, f"Feret: {feret_diam:.2f} um", (text_x, cY - 15), font, 0.6, (255, 255, 255), 2)
                        cv2.putText(calc_mask, f"Circ: {circularity:.4f}", (text_x, cY + 15), font, 0.6, (255, 255, 255), 2)
                    
                    # Add scale info
                    cv2.putText(calc_mask, f"Escala: {micrometers_per_pixel:.2f} um/px", (10, height - 20), font, 0.7, (255, 255, 255), 2)
                    
                    # Save calculation mask
                    calc_mask_path = os.path.join(masks_calc_folder, f"{base_name}-calc{original_extension}")
                    cv2.imwrite(calc_mask_path, calc_mask)
                    
                    # Update progress
                    self.progress_updated.emit(i, overlay_save_path)
                
                except Exception as e:
                    self.error_occurred.emit(f"Error procesando imagen {filename}: {str(e)}")
                    continue
            
            self.processing_done.emit()
        
        except Exception as e:
            self.error_occurred.emit(f"Error general: {str(e)}")


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
        
        self.api_key_input = QLineEdit(self)
        self.api_key_input.setPlaceholderText("Introduce API Key")
        self.api_key_input.setText(self.api_key)
        left_layout.addWidget(self.api_key_input)
        
        self.project_id_input = QLineEdit(self)
        self.project_id_input.setPlaceholderText("Introduce Project ID")
        self.project_id_input.setText(self.project_id)
        left_layout.addWidget(self.project_id_input)
        
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
            images = [f for f in os.listdir(folder) if f.lower().endswith((".tiff", ".tif"))]
            total_images = len(images)
            
            folder_name = os.path.basename(folder)
            self.folder_info_label.setText(f"Carpeta: {folder_name}\nTotal imágenes: {total_images}")
            
            self.progress_bar.setMaximum(total_images)
            self.progress_bar.setValue(0)
            
            # Buscar carpetas overlayed_images existentes
            overlayed_folders = [f for f in os.listdir(folder) if f.startswith("overlayed_images_")]
            
            if overlayed_folders:
                # Ordenar por fecha (la más reciente primero)
                overlayed_folders.sort(reverse=True)
                latest_folder = os.path.join(folder, overlayed_folders[0])
                
                # Cargar las imágenes existentes
                self.image_list.clear()
                overlayed_images = [f for f in os.listdir(latest_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
                
                for image in overlayed_images:
                    image_path = os.path.join(latest_folder, image)
                    self.add_thumbnail(image_path)
                
                # Actualizar la barra de progreso
                self.progress_bar.setValue(len(overlayed_images))
                
                # Actualizar el label con la información de la carpeta procesada
                self.folder_info_label.setText(
                    f"Carpeta: {folder_name}\n"
                    f"Total imágenes: {total_images}\n"
                    f"Procesadas anteriormente: {len(overlayed_images)} (en {overlayed_folders[0]})"
                )
            
            self.btn_process.setEnabled(total_images > 0)
            if total_images == 0:
                self.folder_info_label.setText(f"Carpeta: {folder_name}\nNo hay imágenes para procesar")
    
    def process_images(self):
        if hasattr(self, 'thread') and isinstance(self.thread, ImageProcessorThread) and self.thread.isRunning():
            # Si el proceso está en marcha, lo detenemos
            self.thread.stop()
            self.thread.wait()
            self.btn_process.setText("Procesar Imágenes")
            return
        
        if not self.image_folder:
            return
        
        self.api_key = self.api_key_input.text().strip()
        self.project_id = self.project_id_input.text().strip()
        self.save_settings()
        
        if not self.api_key or not self.project_id:
            self.processed_image_label.setText("Por favor, introduce API Key y Project ID")
            self.original_image_label.setText("Por favor, introduce API Key y Project ID")
            return
        
        self.image_list.clear()
        self.progress_bar.setValue(0)
        
        self.thread = ImageProcessorThread(self.image_folder, self.api_key, self.project_id)
        self.thread.progress_updated.connect(self.update_progress)
        self.thread.processing_done.connect(self.processing_finished)
        self.thread.error_occurred.connect(self.handle_processing_error)  # Nueva conexión para manejar errores
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
            # Obtener la ruta de la imagen procesada
            processed_path = item.data(Qt.ItemDataRole.UserRole)
            self.current_processed_path = processed_path
            
            # Deducir las rutas de las demás imágenes
            filename = os.path.basename(processed_path)
            base_name = os.path.splitext(filename)[0]
            folder_path = os.path.dirname(processed_path)
            parent_folder = os.path.dirname(folder_path)
            
            # Obtener la ruta de la imagen original
            original_filename = filename
            original_path = os.path.join(os.path.dirname(parent_folder), original_filename)
            self.current_original_path = original_path
            
            # Obtener la ruta de la máscara
            mask_folder = os.path.join(parent_folder, "masks")
            extension = os.path.splitext(filename)[1]
            mask_filename = f"{base_name}-mask{extension}"
            mask_path = os.path.join(mask_folder, mask_filename)
            self.current_mask_path = mask_path if os.path.exists(mask_path) else None
            
            # Obtener la ruta de la máscara con cálculos
            mask_calc_folder = os.path.join(parent_folder, "masks_calculations")
            mask_calc_filename = f"{base_name}-calc{extension}"
            mask_calc_path = os.path.join(mask_calc_folder, mask_calc_filename)
            self.current_mask_calc_path = mask_calc_path if os.path.exists(mask_calc_path) else None
            
            # Actualizar el nombre mostrado
            self.image_name_label.setText(original_filename)
            
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
        settings = {"api_key": self.api_key, "project_id": self.project_id}
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f)
    
    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
                self.api_key = settings.get("api_key", "")
                self.project_id = settings.get("project_id", "")
    
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Configurar el estilo de la aplicación
    app.setStyle(QStyleFactory.create("Fusion"))
    
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
