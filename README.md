# Procesador de Imágenes con modelos de Landing.ai

Una aplicación para el procesamiento de imágenes utilizando inteligencia artificial con Landing AI. Esta herramienta ha sido diseñada para entornos de investigación y laboratorio, como parte de un trabajo de fin de grado de la carrera de Farmacia de la universidad CEU Cardenal Herrera.

## Características

- **Procesamiento de imágenes con IA**: Utiliza la API de Landing AI para procesar imágenes y mostrar las predicciones superpuestas.
- **Interfaz adaptativa**: Soporte para modo claro y oscuro, adaptándose automáticamente a la configuración del sistema.
- **Visualización dual**: Permite alternar entre las imágenes originales y procesadas con un sistema de pestañas.
- **Organización automática**: Cada nuevo procesamiento crea nuevas carpetas con marcas de tiempo para evitar sobrescribir procesamientos anteriores.
- **Arrastrar y soltar**: Funcionalidad para arrastrar imágenes directamente al escritorio o a otras aplicaciones.
- **Atajos de teclado**: Mejora de la productividad con atajos para todas las funciones principales.

## Requisitos del sistema

- Python 3.7 o superior
- Conexión a internet (para la API de Landing AI)
- Cuenta en Landing AI con un proyecto configurado
- 4 GB de RAM mínimo (recomendado 8 GB)

## Instalación

### Windows

1. Instala Python desde [python.org](https://www.python.org/downloads/windows/):
   - Durante la instalación, asegúrate de marcar "Add Python to PATH"

2. Abre PowerShell o Símbolo del sistema y ejecuta:
   ```
   pip install PyQt6 Pillow torch numpy
   pip install landingai
   ```

3. Descarga el código de la aplicación:
   ```
   git clone https://github.com/quarac/landingai-image-processor.git
   cd landing-process-images
   ```

4. Ejecuta la aplicación:
   ```
   python landingai-image-processor.py
   ```

### macOS

1. Si no tienes Homebrew, instálalo:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Instala Python usando Homebrew:
   ```
   brew install python
   ```

3. Instala las dependencias:
   ```
   pip3 install PyQt6 Pillow torch numpy
   pip3 install landingai
   ```

4. Descarga el código de la aplicación:
   ```
   git clone https://github.com/quarac/landingai-image-processor.git
   cd landing-process-images
   ```

5. Ejecuta la aplicación:
   ```
   python3 landingai-image-processor.py
   ```

## Guía de uso

1. Inicia la aplicación
2. Introduce tu API Key y Project ID de Landing AI en los campos correspondientes
3. Haz clic en "Seleccionar Carpeta de Imágenes" y elige la carpeta con las imágenes a procesar
4. Haz clic en "Procesar Imágenes" para comenzar el procesamiento
5. Una vez procesadas, las imágenes aparecerán como miniaturas en la lista
6. Haz clic en cualquier miniatura para ver la imagen en grande
7. Utiliza las pestañas para alternar entre la versión procesada y la original
8. Puedes arrastrar cualquier imagen desde la aplicación a otra ubicación

## Atajos de teclado

- **Ctrl+O**: Seleccionar carpeta de imágenes
- **Ctrl+P**: Iniciar/detener procesamiento
- **Ctrl+I**: Alternar entre imagen procesada y original

## Solución de problemas

- Si la aplicación no muestra iconos en los botones, esto es normal en Windows y no afecta la funcionalidad.
- En caso de errores relacionados con la API, verifica tu conexión a internet y las credenciales de Landing AI.
- Si la aplicación no inicia, asegúrate de tener instaladas todas las dependencias.

## Licencia

Este proyecto está bajo la Licencia Pública General GNU (GPL-3.0). Puedes ver el texto completo de la licencia en el archivo [LICENSE](LICENSE).

### Resumen de la licencia GPL-3.0

La Licencia Pública General GNU (GPL) garantiza tu derecho a ejecutar, estudiar, compartir y modificar el software. Pero si distribuyes el software o sus modificaciones, debes hacerlo bajo los mismos términos de la GPL. Esto significa que cualquier software derivado también debe ser de código abierto y estar bajo la misma licencia.

Para más detalles, consulta el archivo [LICENSE](LICENSE) o visita el sitio web oficial de la [Licencia GPL](https://www.gnu.org/licenses/gpl-3.0.html).

## Créditos

Desarrollado por Abelardo Sánchez Ribera.

Utiliza la API de [Landing AI](https://landing.ai/) para el procesamiento de imágenes. 
