import sys
import os
import torch
import json
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QListWidget, QListWidgetItem, QHBoxLayout, QLineEdit, QSplitter, QProgressBar, QSizePolicy, QTabWidget, QStyleFactory)
from PyQt6.QtGui import QPixmap, QIcon, QDrag, QFontDatabase, QFont, QPalette, QColor, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QMimeData
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions

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
        
        # Create base folder name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_folder = os.path.join(self.image_folder, f"overlayed_images_{timestamp}")
        
        # Handle duplicates by adding (n) if needed
        folder_name = base_folder
        counter = 1
        while os.path.exists(folder_name):
            folder_name = f"{base_folder}({counter})"
            counter += 1
        
        os.makedirs(folder_name, exist_ok=True)
        
        images = [f for f in os.listdir(self.image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        total_images = len(images)
        
        predictor = Predictor(endpoint_id=self.project_id, api_key=self.api_key)
        
        for i, filename in enumerate(images, start=1):
            if self.should_stop:
                break
                
            image_path = os.path.join(self.image_folder, filename)
            img = Image.open(image_path).convert("RGB")
            predictions = predictor.predict(img)
            overlayed_img = overlay_predictions(predictions, img)
            save_path = os.path.join(folder_name, filename)
            overlayed_img.save(save_path)
            
            self.progress_updated.emit(i, save_path)
        
        self.processing_done.emit()

class ImageProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_folder = None
        self.api_key = ""
        self.project_id = ""
        self.current_image_path = None
        self.current_original_path = None  # Ruta de la imagen original
        self.current_processed_path = None  # Ruta de la imagen procesada
        
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
        
        # Añadir las pestañas al selector
        self.image_type_selector.addTab(self.processed_tab, "Imagen Procesada")
        self.image_type_selector.addTab(self.original_tab, "Imagen Original")
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
            images = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
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
        self.thread.start()
        
        self.btn_process.setText("Detener proceso")
    
    def update_progress(self, current_image, image_path):
        self.progress_bar.setValue(current_image)
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
            processed_path = item.data(Qt.ItemDataRole.UserRole)
            self.current_processed_path = processed_path
            
            # Obtener la ruta de la imagen original
            original_filename = os.path.basename(processed_path)
            original_path = os.path.join(self.image_folder, original_filename)
            self.current_original_path = original_path
            
            # Actualizar la imagen según el tipo seleccionado
            self.on_tab_changed()
            
            # Actualizar el nombre mostrado
            self.image_name_label.setText(original_filename)
    
    def on_tab_changed(self):
        if self.image_type_selector.currentIndex() == 1:  # Pestaña "Imagen Original"
            self.current_image_path = self.current_original_path
        else:  # Pestaña "Imagen Procesada"
            self.current_image_path = self.current_processed_path
        
        self.update_large_image()
    
    def update_large_image(self):
        if not self.current_image_path:
            return
            
        original_pixmap = QPixmap(self.current_image_path)
        # Usar el label activo según la pestaña seleccionada
        active_label = self.processed_image_label if self.image_type_selector.currentIndex() == 0 else self.original_image_label
        label_size = active_label.size()
        
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
        active_label.setPixmap(pixmap)
    
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Configurar el estilo de la aplicación
    app.setStyle(QStyleFactory.create("Fusion"))
    
    window = ImageProcessorApp()
    window.show()
    sys.exit(app.exec())
