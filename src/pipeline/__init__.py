"""
Imagery Processing Pipeline

Three-stage async pipeline:
1. Rembg - Background removal
2. RealESRGAN - 4K tiled upscaling
3. Nano Banana - Smart placement API
"""

