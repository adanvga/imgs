import os
import cv2
import csv
import shutil
import time
import numpy as np
from copy import deepcopy
from datetime import datetime, timedelta
from ultralytics import YOLO
from vehicle import Car
from controller import Display, Keyboard
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import sys
# recibe una imagen RGB necesariamente
def img_preprocess(image):
    h, w = image.shape[:2]
    # Res: 120x300
    # 55% -> 66 px
    # 66.9% -> 200 px
    crop_h_start = int(h * 0.6) # 55% vertical
    crop_w = int(w * 1)       # 67% horizontal
    crop_w_start = (w - crop_w) // 2
    image = image[h-crop_h_start:, crop_w_start:crop_w_start + crop_w]
    #image = cv2.Canny(image, 75, 150)
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  
    image = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.resize(image, (200, 66))
    return image.astype(np.float32) / 255.0

def predecir_angulo(imagen, modelo):
    img_prep = img_preprocess(imagen)
    img_prep = np.expand_dims(img_prep, axis=0)
    try:
        angulo = modelo.predict(img_prep, verbose=0)[0][0]
        return angulo
    except Exception as e:
        print(f"[ERROR] model.predict: {e}")
        return float('inf')

PAR_HOUGH_RHO=1
PAR_HOUGH_THETA=np.pi/180
PAR_HOUGH_THRESHOLD=5
PAR_HOUGH_MINLINELEN=1
PAR_HOUGH_MAXLINEGAP=2

angle = 0.0
speed = 15
controlled_speed = 33

model_for_object_detection = YOLO("yolov8n.pt")
model_dir = r"C:\Users\Adan\Documents\Tmp\Final"
name_best_model = rf"{model_dir}\model_best.keras"
name_regl_model = rf"{model_dir}\model_nvidia.keras"
behavioural_model_name = name_regl_model


def get_image(_camera):
    raw_image = _camera.getImage()  
    _image = np.frombuffer(raw_image, np.uint8).reshape(
        (_camera.getHeight(), _camera.getWidth(), 4)
    )
    _image = _image[:, :, :3]
    return _image

def display_color_image(display, image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

def generar_puntos_trapecio(imagen, base_B, tapa_A, altura_H,offset_H=0):
    alto, ancho = imagen.shape[:2]
    centro_x = ancho // 2
    centro_x += offset_H
    puntos = np.array([[
        (centro_x - tapa_A // 2, alto - altura_H),  # esquina superior izquierda
        (centro_x + tapa_A // 2, alto - altura_H),  # esquina superior derecha
        (centro_x + base_B // 2, alto),             # esquina inferior derecha
        (centro_x - base_B // 2, alto)              # esquina inferior izquierda
    ]], dtype=np.int32)

    return puntos

def aplicar_mascara_trapecio(imagen, puntos):
    mascara = np.zeros_like(imagen)
    color_mascara = (255,) * imagen.shape[2] if len(imagen.shape) == 3 else 255
    cv2.fillPoly(mascara, puntos, color_mascara)
    imagen_mascara = cv2.bitwise_and(imagen, mascara)
    return imagen_mascara

def dibujar_trapecio(imagen, puntos, color=(0, 0, 250), thickness=1):
    imagen_resultado = imagen.copy()
    cv2.polylines(imagen_resultado, puntos, isClosed=True, color=color, thickness=thickness)
    return imagen_resultado

def preparar_imagen(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.GaussianBlur(imagen, (5, 5), 1.5)
    imagen = cv2.Canny(imagen, 75, 150)
    return imagen

def calcular_centrado_horizontal(lineas,ancho,tolerancia=0.05):
    """
    Calcula un valor entre -1 y 1 que indica qué tan centradas están
    las líneas detectadas por Hough respecto al centro horizontal de la imagen.
    
    -1 → todas las líneas a la izquierda
     0 → líneas centradas
    +1 → todas las líneas a la derecha
    """
    # Si no se encuentran líneas, en línea recta
    if lineas is None or len(lineas) == 0:
        return 0.0
    centro_imagen = ancho*0.5 # Centro de la imagen
    centro_imagen = ancho*0.17
    desviaciones = []
    for linea in lineas:
        x1, _, x2, _ = linea[0]
        centro_linea = (x1 + x2) / 2 # Centro de linea 
        desviacion = (centro_linea - centro_imagen) / (ancho / 2)  # Normalizado [-1, 1]
        desviaciones.append(desviacion) # Agregar para el promedio 
    promedio = np.mean(desviaciones)
    if abs(promedio) < tolerancia:
        promedio = 0.0 # Dentro de la tolerancia conducimos derecho 
    return round(promedio,4)

def calcular_error_carril(lineas, ancho_img, tolerancia=0.05):
    """
    Calcula el error de centrado con respecto al centro del carril.
    Usa líneas a la izquierda y derecha de la imagen para determinar el centro del carril.
    
    Devuelve un valor entre -1 y 1:
    -1 → estás muy a la derecha (corregir a la izquierda)
     0 → centrado
    +1 → estás muy a la izquierda (corregir a la derecha)
    """
    if lineas is None or len(lineas) == 0:
        return 0.0  # sin líneas visibles, conducir recto

    lineas_izquierda = []
    lineas_derecha = []

    for linea in lineas:
        x1, _, x2, _ = linea[0]
        x_centro = (x1 + x2) / 2

        if x_centro < ancho_img * 0.4:
            lineas_izquierda.append(x_centro)
        elif x_centro > ancho_img * 0.6:
            lineas_derecha.append(x_centro)

    if not lineas_izquierda or not lineas_derecha:
        return 0.0  # sin suficiente información, conducir recto

    izquierda = np.mean(lineas_izquierda)
    derecha = np.mean(lineas_derecha)
    centro_carril = (izquierda + derecha) / 2
    centro_imagen = ancho_img / 2

    error = (centro_carril - centro_imagen) / (ancho_img / 2)

    if abs(error) < tolerancia:
        return 0.0

    return round(error, 4)




def control_pd_simple(actual, anterior=None, kp=1.0, kd=0.0):
    p = actual # Proporcional
    d = 0 if anterior != None else  actual - anterior # Derivada
    salida = kp * p + kd * d # Salida 
    return float(salida)

def detectar_objetos(frame):
    results = model_for_object_detection(frame, verbose=False)[0]
    resumen = {
        "peatones": 0,
        "vehiculos": 0
    }
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # Persona
            resumen["peatones"] += 1
        elif cls in {2, 3, 5, 7}:  # Coche, moto, bus, camión (según COCO)
            resumen["vehiculos"] += 1
    return resumen

def generar_imagen_vehicle_data(_speed, _angle, _brake, _entorno, _dist_closest_obj, _actual_speed, 
                                 ancho=320, alto=135):
    img = np.zeros((alto, ancho, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.6
    grosor = 1
    blanco = (255, 255, 255)
    y = 20
    dy = 20

    # Texto principal
    cv2.putText(img, f"Vel. targ: {_speed:.1f} km/h", (10, y), font, escala, blanco, grosor); y += dy
    cv2.putText(img, f"Vel. real: {_actual_speed:.1f} km/h", (10, y), font, escala, blanco, grosor); y += dy
    cv2.putText(img, f"Angulo: {_angle:.5f} rad", (10, y), font, escala, blanco, grosor); y += dy
    cv2.putText(img, f"Peatones: {_entorno['peatones']}", (10, y), font, escala, blanco, grosor); y += dy
    cv2.putText(img, f"Vehiculos: {_entorno['vehiculos']}", (10, y), font, escala, blanco, grosor); y += dy
    cv2.putText(img, f"Dist. obstaculo: {_dist_closest_obj:.2f} m", (10, y), font, escala, blanco, grosor); y += dy

    # === Pedal de freno visual ===
    # Posición y tamaño del pedal
    x0 = 10 + int(ancho*0.75) + 10  # alineado al texto, con margen
    y0 = y - 4*dy             # y actual ya fue incrementado justo después del texto
    x1 = x0 + int(ancho*0.15)
    y1 = y0 + int(3*dy)

    # Color rojo proporcional a _brake
    intensidad = int(_brake * 255)
    color_freno = (0, 0, intensidad)  # BGR: rojo puro

    # Dibujar rectángulo con esquinas redondeadas (simulación)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color_freno, -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Contorno blanco para el pedal
    cv2.rectangle(img, (x0, y0), (x1, y1), blanco, 1, cv2.LINE_AA)

    # Texto interno
    cv2.putText(img, "BRAKE", (x0 + 3, y1 - 10), font, 0.4, blanco, 1, cv2.LINE_AA)

    return img

def main():
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    record_training_data=False
    conducimos_por_modelo = False
    conducimos_por_crtpid = not(conducimos_por_modelo)
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    robot = Car()
    timestep = int(robot.getBasicTimeStep())

    camara_frontal = robot.getDevice("camera")
    camara_frontal.enable(timestep)

    camera_left = None
    camera_right = None

    #camera_left = robot.getDevice("camera_left")
    #camera_right = robot.getDevice("camera_right")
    if camera_left != None:
        camera_left.enable(timestep)
    if camera_right != None:
        camera_right.enable(timestep)

    camera_high = robot.getDevice("high_cam")
    camera_high.enable(timestep)

    lidar = robot.getDevice("lidar")
    if lidar != None:
        lidar.enable(timestep)

    gps = robot.getDevice("gps")
    gps.enable(timestep)

    keyboard=Keyboard()
    keyboard.enable(timestep)

    display_img = Display("display_image")
    display_veh_data = Display("display_veh_data")

    speed = controlled_speed
    angle = 0
    brake = 0
    
    adj_speed = 2
    adj_angle = 0.18
    adj_angle = 0.36
    angle_anterior = 0.0

    ultimo_t_tecla              = datetime.min
    intrvl_tecla                = timedelta(milliseconds=200)

    ultimo_t_giro_manual        = datetime.min
    intrvl_giro_manual          = timedelta(milliseconds=200)

    angle_vuelta_manual = 0.25
    ultimo_t_vuelta_manual = datetime.min
    intrvl_vuelta_manual = timedelta(milliseconds=1700)

    fps_sensor_detection = 100
    intrvl_sensor_detection      = timedelta(seconds=1 / fps_sensor_detection)
    ultimo_t_sensor_detection    = datetime.min

    fps_observacion_frontal = 120
    intrvl_observacion_frontal  = timedelta(seconds=1 / fps_observacion_frontal)
    ultimo_t_observacion_frontal= datetime.min

    fps_object_detection = 6
    intrvl_deteccion_obj        = timedelta(seconds=1 / fps_object_detection)
    ultimo_t_deteccion_obj      = datetime.min
    ultimo_res_deteccion_obj = {
        "peatones": 0,
        "vehiculos": 0,
    }

    ultima_posicion_gps = [0.0, 0.0, 0.0]
    ultimo_tiempo_gps = robot.getTime()
    vel_real_actual_kmh = 0.0

    fps_curvas_reg = 6      # fps_curvas regulares (por ejemplo, en curva)
    fps_curvas_str = 16      # fps_curvas fueretes  (por ejemplo, en curva)
    fps_linea_recta = 1  # fps_curvas reducido cuando vamos recto
    intrvl_captura_imgs         = timedelta(seconds=1.0 / fps_curvas_reg)

    fps_observ_conduccion = 1.5
    intrvl_observ_conduccion    = timedelta(seconds=1.0 / fps_observ_conduccion)
    ultimo_t_captura_imgs       = datetime.min
    ultimo_t_observ_conduccion  = datetime.min

    modelo_propio_behavioural_cloning = load_model(behavioural_model_name)
    modelo_propio_behavioural_cloning.summary()
    modelo = modelo_propio_behavioural_cloning

    print("🔢 Total de parámetros     :", modelo.count_params())
    print("🧠 Parámetros entrenables :", sum([K.count_params(w) for w in modelo.trainable_weights]))
    print("🧊 Parámetros no entrenables:", sum([K.count_params(w) for w in modelo.non_trainable_weights]))

    distancia_obj_mas_cercano = float('inf')
    distancia_anterior_omc = float('inf')

    print(f"Starting at {speed} km/h")

    while robot.step() != -1:
        ahora = datetime.now()
        img_vista_actual = get_image(camara_frontal)
        # Calcular velocidad real actual
        pos = gps.getValues()  # [x, y, z]
        t = robot.getTime()
        dt = t - ultimo_tiempo_gps
        if dt > 0:
            dx = pos[0] - ultima_posicion_gps[0]
            dy = pos[1] - ultima_posicion_gps[1]
            dz = pos[2] - ultima_posicion_gps[2]
            distancia = (dx**2 + dy**2 + dz**2)**0.5
            vel_real_actual_kmh = (distancia / dt) * 3.6  # m/s → km/h
        ultima_posicion_gps = pos[:]  # o list(pos)
        ultimo_tiempo_gps = t

        display_color_image(display_veh_data, generar_imagen_vehicle_data(speed, angle, brake, ultimo_res_deteccion_obj,distancia_obj_mas_cercano,vel_real_actual_kmh))

        # Detección de objetos con CNN preentrenada
        if (ahora - ultimo_t_deteccion_obj) >= intrvl_deteccion_obj:
            image_high = get_image(camera_high)
            ultimo_res_deteccion_obj = detectar_objetos(image_high)
            ultimo_t_deteccion_obj = ahora

        # Deteccion con radar
        if (ahora - ultimo_t_sensor_detection) >= intrvl_sensor_detection and lidar != None:
            ranges = lidar.getRangeImage()
            distancia_obj_mas_cercano = ranges[len(ranges)//2]
            centro = len(ranges) // 2
            n = 5
            distancia_obj_mas_cercano  = min(ranges[centro - n : centro + n + 1])

        # Frenado
        if True:
            distancia_segura = 15.0
            umbral_frenado = 7.0
            if distancia_obj_mas_cercano < distancia_segura:
                error_distancia = distancia_obj_mas_cercano - umbral_frenado
                velocidad_objetivo = control_pd_simple(error_distancia, distancia_anterior_omc, kp=0.8)
                velocidad_objetivo = max(0, min(controlled_speed, velocidad_objetivo))
                speed = velocidad_objetivo
                error_freno = umbral_frenado - distancia_obj_mas_cercano
                error_freno = max(0.0, min(1.0, error_freno))
                brake = min(1.0, error_freno * 0.7)
            elif distancia_obj_mas_cercano < umbral_frenado:
                brake = 1
                speed = 0
            elif distancia_obj_mas_cercano == float('inf'):
                brake = 0.0
                speed = max(controlled_speed,speed)
            distancia_anterior_omc = distancia_obj_mas_cercano

        # Volante controlado por CNN
        if (ahora - ultimo_t_observ_conduccion) >= intrvl_observ_conduccion and \
           (ahora - ultimo_t_giro_manual) >= intrvl_giro_manual and \
           (ahora - ultimo_t_vuelta_manual) >= intrvl_vuelta_manual and \
            conducimos_por_modelo:
            start = time.perf_counter()
            angle = adn.predecir_angulo(img_vista_actual,modelo_propio_behavioural_cloning)
            elapsed = (time.perf_counter() - start) * 1000  # ms

        # Establecemos los parámetros al final del bucle
        robot.setSteeringAngle(np.clip(angle, -0.8, 0.8))
        robot.setBrakeIntensity(brake)
        robot.setCruisingSpeed(speed)

    print("")
    print("")
    print("")
    print("autodrive: exit")
    print("")
    print("")
    print("")

if __name__ == "__main__":
    main()