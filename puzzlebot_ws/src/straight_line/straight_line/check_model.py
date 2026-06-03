#!/usr/bin/env python3
"""
check_model.py — Herramienta de diagnóstico para best_copy.onnx
================================================================
Ejecutar FUERA de ROS:
    python3 check_model.py [ruta_al_modelo] [ruta_imagen_opcional]

Muestra:
  • Shape de entrada y salida del modelo
  • Nombres de clases embebidos en los metadatos (si existen)
  • Si se da una imagen: top-5 detecciones con score y clase
  • Formato detectado (YOLOv5 vs YOLOv8)
"""

import sys
import os
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print('ERROR: pip install onnxruntime')
    sys.exit(1)

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/ubuntu/puzzlebot_docker/puzzlebot_ws/models/best_copy.onnx'

IMAGE_PATH = sys.argv[2] if len(sys.argv) > 2 else None

# ── Cargar modelo ─────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'Modelo: {MODEL_PATH}')
print(f'{"="*60}')

sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

# ── Metadatos ─────────────────────────────────────────────────────────────────
meta = sess.get_modelmeta()
print(f'\nProducer   : {meta.producer_name}')
print(f'Domain     : {meta.domain}')
print(f'Description: {meta.description}')
print(f'Version    : {meta.version}')

if meta.custom_metadata_map:
    print('\nMetadatos personalizados:')
    for k, v in meta.custom_metadata_map.items():
        print(f'  {k} = {v}')
    # Muchos modelos YOLO guardan "names" como dict string
    if 'names' in meta.custom_metadata_map:
        import ast
        try:
            names = ast.literal_eval(meta.custom_metadata_map['names'])
            print(f'\n>>> CLASES DEL MODELO (desde metadatos):')
            for idx, name in (names.items() if isinstance(names, dict) else enumerate(names)):
                print(f'    {idx}: {name}')
        except Exception as e:
            print(f'  (no se pudo parsear "names": {e})')

# ── Inputs / Outputs ──────────────────────────────────────────────────────────
print('\nENTRADAS:')
for inp in sess.get_inputs():
    print(f'  {inp.name}  shape={inp.shape}  type={inp.type}')

print('\nSALIDAS:')
for out in sess.get_outputs():
    print(f'  {out.name}  shape={out.shape}  type={out.type}')

# Detectar formato
out0_shape = sess.get_outputs()[0].shape
is_v8 = (len(out0_shape) == 3 and
         isinstance(out0_shape[1], int) and isinstance(out0_shape[2], int) and
         out0_shape[1] < out0_shape[2])

print(f'\nFormato detectado: {"YOLOv8 [1, 4+nc, anchors]" if is_v8 else "YOLOv5 [1, anchors, 5+nc]"}')

# Número de clases
if is_v8:
    nc = out0_shape[1] - 4 if isinstance(out0_shape[1], int) else '?'
else:
    nc = out0_shape[2] - 5 if isinstance(out0_shape[2], int) else '?'
print(f'Clases (nc): {nc}')

# ── Inferencia de prueba ──────────────────────────────────────────────────────
if IMAGE_PATH and os.path.exists(IMAGE_PATH):
    print(f'\n{"="*60}')
    print(f'Inferencia sobre: {IMAGE_PATH}')
    print(f'{"="*60}')

    img  = cv2.imread(IMAGE_PATH)
    h, w = img.shape[:2]
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    sz = inp_shape[2] if isinstance(inp_shape[2], int) else 640

    resized = cv2.resize(img, (sz, sz))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob    = np.expand_dims(rgb.transpose(2, 0, 1), 0)

    out = sess.run(None, {inp_name: blob})[0]
    print(f'Output shape: {out.shape}')

    if is_v8:
        pred     = out[0].T          # [anchors, 4+nc]
        cls_sc   = pred[:, 4:]
        scores   = cls_sc.max(axis=1)
        cls_ids  = cls_sc.argmax(axis=1)
    else:
        pred     = out[0]            # [anchors, 5+nc]
        obj      = pred[:, 4]
        cls_sc   = pred[:, 5:]
        scores   = obj * cls_sc.max(axis=1)
        cls_ids  = cls_sc.argmax(axis=1)

    top5_idx = scores.argsort()[::-1][:5]
    print('\nTop-5 detecciones (sin NMS):')
    print(f'  {"Rank":<5} {"Score":>7}  {"ClaseIdx":>8}  {"cx_norm":>8}  {"cy_norm":>8}')
    for rank, i in enumerate(top5_idx):
        cx = float(pred[i, 0]) / sz
        cy = float(pred[i, 1]) / sz
        print(f'  {rank+1:<5} {scores[i]:>7.4f}  {cls_ids[i]:>8}  {cx:>8.3f}  {cy:>8.3f}')

    print('\n>>> Edita CLASS_NAMES en sign_detector.py con el orden correcto de índices.')
else:
    print('\nTip: pasa una imagen como segundo argumento para ver detecciones reales:')
    print(f'  python3 check_model.py {MODEL_PATH} /ruta/imagen.jpg')

print(f'\n{"="*60}\n')
