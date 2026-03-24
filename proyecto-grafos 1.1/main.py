import cv2
import math
import os
import urllib.request
import numpy as np

# ==========================================
# 1. DESCARGA DEL MODELO DE INTELIGENCIA ARTIFICIAL
# ==========================================
# MediaPipe Tasks API requiere un modelo ya entrenado para identificar las manos.
# Aquí indicamos la URL oficial del modelo y la ruta local donde se guardará.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

# Si el archivo del modelo no existe en nuestra carpeta local, lo descargamos automáticamente.
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo de MediaPipe (hand_landmarker.task)... Esto solo tomara unos segundos la primera vez.")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 2. CONSTANTES DE CONFIGURACIÓN Y SENSIBILIDAD
# ==========================================
PINCH_THRESHOLD = 70  # Distancia (en píxeles) máxima entre pulgar e índice para considerar que haces "pinza".
NODE_RADIUS = 20      # El tamaño (radio) de los puntos/nodos dibujados en la pantalla.
HOLD_RADIUS = 70      # Tolerancia de movimiento: si tu dedo se mueve dentro de este radio, el programa aún lo considera "quieto".
FRAMES_TO_CREATE = 90 # Número de fotogramas (frames) que debes mantener quieto el dedo para crear un nodo (~3 segundos a 30fps).
FRAMES_TO_DELETE = 90 # Número de fotogramas que debes mantener el puño cerrado para borrar (~3 segundos a 30fps).
DELETE_RADIUS = 50    # Distancia de búsqueda máxima desde tu puño para encontrar y borrar un nodo/arista.

# ==========================================
# 3. CONFIGURACIÓN DEL DETECTOR DE MANOS (API V2)
# ==========================================
# Configuramos el archivo base donde está nuestro modelo previamente descargado.
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
# Configuramos las opciones de seguimiento (1 mano, 80% de confianza para mayor precisión).
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8,
    min_hand_presence_confidence=0.8,
    min_tracking_confidence=0.8,
)
# Creamos la instancia del detector de manos usando las opciones definidos arriba.
detector = vision.HandLandmarker.create_from_options(options)

# ==========================================
# 4. ESTRUCTURAS DE DATOS DE LA GRÁFICA MATEMÁTICA
# ==========================================
nodes = [] # Lista de nodos. Cada nodo es un diccionario: {'id': número, 'x': pos_x, 'y': pos_y, 'color': (B,G,R)}
edges = set() # Usamos un 'set' (conjunto) para evitar aristas duplicadas. Guarda tuplas de ejemplo: (1, 2)
next_node_id = 1 # ID numérico secuencial que se asignará al próximo nodo que se cree.

# ==========================================
# 5. FUNCIONES AUXILIARES MATEMÁTICAS
# ==========================================

def dist(p1, p2):
    """Calcula la distancia directa (hipotenusa) entre dos puntos (x, y)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def get_closest_node(pos, max_dist=HOLD_RADIUS):
    """Busca y devuelve el ID del nodo que esté más cerca de la posición 'pos',
    siempre y cuando esté dentro de la distancia 'max_dist'."""
    closest = None
    min_d = max_dist
    for node in nodes:
        d = dist(pos, (node['x'], node['y']))
        if d < min_d:
            min_d = d
            closest = node['id']
    return closest

def get_node_by_id(node_id):
    """Busca en la lista de nodos aquel que tenga el ID especificado y lo devuelve."""
    for node in nodes:
        if node['id'] == node_id:
            return node
    return None

def point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Calcula la distancia más corta desde un punto (px, py) a un segmento de línea (la conexión de un nodo a otro)
    Útil para saber si quieres borrar directamente una linea sin borrar su nodo originador."""
    line_mag = dist((x1, y1), (x2, y2))
    if line_mag < 0.000001:
        return dist((px, py), (x1, y1))

    # Matemática vectorial para proyectar nuestro punto sobre la línea finita
    u = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1))) / (line_mag * line_mag)
    if u < 0.0 or u > 1.0:
        # El punto está más allá de los extremos de la línea, medir a los extremos
        d1 = dist((px, py), (x1, y1))
        d2 = dist((px, py), (x2, y2))
        return min(d1, d2)
    else:
        # El punto cae dentro del segmento
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        return dist((px, py), (ix, iy))

def check_is_fist(hand_landmarks, w, h):
    """
    Verifica si la mano está en forma de puño comprobando la biomecánica:
    si la "punta" de los dedos está más cerca de la muñeca que sus propios nudillos, la mano está cerrada.
    """
    wrist = (hand_landmarks[0].x * w, hand_landmarks[0].y * h) # Muñeca (punto 0 en el modelo 3D)
    fingers = [(8, 5), (12, 9), (16, 13), (20, 17)] # (Tupla con: Punta del dedo, Nudillo de ese dedo)
    for tip_idx, mcp_idx in fingers:
        tip = (hand_landmarks[tip_idx].x * w, hand_landmarks[tip_idx].y * h)
        mcp = (hand_landmarks[mcp_idx].x * w, hand_landmarks[mcp_idx].y * h)
        # Si tan solo uno de los dedos NO está doblado (punta más lejos que el nudillo), entonces no es un puño.
        if dist(tip, wrist) > dist(mcp, wrist):
            return False
    return True # Todos los dedos cumplieron la condición, ¡es un puño!

# ==========================================
# 6. CICLO PRINCIPAL (CAPTURA DE VIDEO Y LOGICA DE LA APP)
# ==========================================
def main():
    global next_node_id, nodes, edges, detector

    # --- Variables para recordar tiempos y estados a través de los fotogramas ---
    last_index_pos = None        # Recuerda dónde estaba tu dedo índice hace un momento
    index_still_frames = 0       # Contador de tiempo (marcos) que llevas quieto el índice (para crear nodos)
    dragging_from_node_id = None # Si estás arrastrando una línea con pinza, guarda de qué ID de nodo provienes
    pinch_active = False         # Variable interruptor "booleana" que recuerda si tienes la pinza cerrada

    last_fist_pos = None         # Recuerda dónde estaba tu puño hace un momento
    fist_still_frames = 0        # Contador de tiempo que llevas quieto el puño (para eliminar cosas)
    
    # --- Estabilizadores Matemáticos (Filtro EMA) ---
    # Esto elimina que tu mano o el sensor tiemblen un poco o tengan "ruido" dándote extrema fluidez
    smooth_index = None  
    smooth_thumb = None
    smooth_center = None
    ALPHA = 0.5 # Qué tanto confiamos en la posición nueva (movimiento rapido) vs la vieja (0 a 1)

    # Abrir cámara web y crear ventana
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Grafo AR', cv2.WINDOW_NORMAL)
    
    # Manejadores de texto temporal de interfaz grafica (Ej. "Nodo creado")
    message_text = ""
    message_frames = 0
    
    # Este 'while' correrá eternamente mientras la camara siga mandando video y no presiones Q
    while cap.isOpened():
        ret, frame = cap.read() # Leer cada "fotograma" individual de la cámara web
        if not ret: break
            
        # Efecto espejo (flip horizontal), para que tu brazo derecha se vea a la derecha 
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # CV2 usa formato BGR colorido invertido, pero MediaPipe pide formato RGB, así que se convierte
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # MANDAMOS LA IMAGEN A LA INTELIGENCIA ARTIFICIAL A QUE ENCUENTRE MANOS
        detection_result = detector.detect(mp_image)
        
        # Variables locales reseteadas cada ciclo que contendran lo que encontremos hoy en los dedos
        index_pos = None
        thumb_pos = None
        hand_center = None
        is_fist = False
        
        # === ¿SE ENCONTRÓ ALGUNA MANO EN EL VIDEO? ===
        if detection_result.hand_landmarks:
            hand_landmarks = detection_result.hand_landmarks[0] # Usaremos la primera mano que detecte el modelo
            
            # Obtener puntos de interés, convirtiendo las propopciones (0 a 1.0) a píxeles exactos (multiplicando x ancho W y x alto H)
            hx, hy = int(hand_landmarks[9].x * w), int(hand_landmarks[9].y * h) # Nudillo del dedo medio asume el Centro de la Mano
            ix, iy = int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h) # Punta exacta del Dedo Índice
            tx, ty = int(hand_landmarks[4].x * w), int(hand_landmarks[4].y * h) # Punta exacta del Dedo Pulgar
            
            # Aplicamos el famoso Suavizado EMA que implementamos
            if smooth_center is None: smooth_center = (hx, hy)
            else: smooth_center = (int(ALPHA * hx + (1-ALPHA) * smooth_center[0]), int(ALPHA * hy + (1-ALPHA) * smooth_center[1]))
            
            if smooth_index is None: smooth_index = (ix, iy)
            else: smooth_index = (int(ALPHA * ix + (1-ALPHA) * smooth_index[0]), int(ALPHA * iy + (1-ALPHA) * smooth_index[1]))
                
            if smooth_thumb is None: smooth_thumb = (tx, ty)
            else: smooth_thumb = (int(ALPHA * tx + (1-ALPHA) * smooth_thumb[0]), int(ALPHA * ty + (1-ALPHA) * smooth_thumb[1]))

            hand_center = smooth_center
            index_pos = smooth_index
            thumb_pos = smooth_thumb

            # Usar la función heurística (verificación biomecánica) para saber si la mano es un puño
            is_fist = check_is_fist(hand_landmarks, w, h)
            
            if not is_fist:
                # Si estamos interactuando normal, pintar puntitos rojo y azul en índice/pulgar de la app
                cv2.circle(frame, index_pos, 8, (255, 0, 0), -1) 
                cv2.circle(frame, thumb_pos, 8, (0, 0, 255), -1) 
            else:
                # Si tenemos el puño cerrado, poner solo un gran punto rojo justo al centro del puño
                cv2.circle(frame, hand_center, 15, (0, 0, 255), -1)

        # ========================================================
        # 7. LÓGICA DE NEGOCIO (DETERMINAR QUÉ GESTO HACE EL HUMANO)
        # ========================================================
        
        if is_fist and hand_center:
            # ---> CASO A: GESTO DETECTADO = PUÑO (INTENCIÓN DE BORRAR) 
            
            if last_fist_pos is None:
                last_fist_pos = hand_center
                fist_still_frames = 1
            elif dist(hand_center, last_fist_pos) < HOLD_RADIUS:
                # El puño sigue prácticamente en el mismo lugar de tolerancia que antes
                fist_still_frames += 1 # Sumar el tiempo
                progress = min(1.0, fist_still_frames / FRAMES_TO_DELETE) # Matematica para la barra de carga visual
                
                # Feedback visual en pantalla: un anillo rojo grueso creciendo simulando un tiempo de carga 
                cv2.circle(frame, last_fist_pos, int(DELETE_RADIUS * progress), (0, 0, 255), 3)
                
                # ¡SE CUMPLIERON LOS 3 SEGUNDOS ESPERADOS! PROCEDER A BORRAR
                if fist_still_frames >= FRAMES_TO_DELETE:
                    deleted_something = False
                    
                    # Logica del borrado 1: Intentar borrar un NODO cercno al lugar
                    closest_id = get_closest_node(last_fist_pos, DELETE_RADIUS)
                    if closest_id is not None:
                        # Si borramos nodo, purgar toda la matriz de nodos y borrar sus aristas asociadas
                        nodes = [n for n in nodes if n['id'] != closest_id]
                        edges = {e for e in edges if closest_id not in e}
                        message_text = f"Nodo {closest_id} borrado"
                        deleted_something = True
                    else:
                        # Lógica del borrado 2: Si no había nodo, quizá quiere borrar una LÍNEA/ARISTA vacia directamente
                        closest_edge = None
                        min_edge_dist = DELETE_RADIUS
                        for edge in list(edges): # Buscamos en todas las lineas activas en pantalla
                            n1 = get_node_by_id(edge[0])
                            n2 = get_node_by_id(edge[1])
                            if n1 and n2:
                                d = point_to_segment_dist(last_fist_pos[0], last_fist_pos[1], n1['x'], n1['y'], n2['x'], n2['y'])
                                if d < min_edge_dist: # Buscamos a la línea numéricamente más cercana al puño
                                    min_edge_dist = d
                                    closest_edge = edge
                        if closest_edge:
                            edges.remove(closest_edge)
                            message_text = "Arista borrada"
                            deleted_something = True
                    
                    if deleted_something:
                        message_frames = 60 # Mostrar el mensaje temporal durante dos segundos
                    
                    fist_still_frames = 0 # Reiniciar progreso por si se deja el puño
            else:
                last_fist_pos = hand_center # Movió el puño a otro lado, arrancar el conteo de nuevo
                fist_still_frames = 0
            
            # Como su intención firme es borrar, apagamos cualquier conexión/creación fantasma activa
            pinch_active = False
            dragging_from_node_id = None
            index_still_frames = 0
            last_index_pos = None
            
        elif index_pos and thumb_pos:
            # ---> CASO B: GESTO DETECTADO = MANO ABIERTA NORMAL 
            
            last_fist_pos = None # Reiniciamos cualquier cuenta atrás de borrar puños
            fist_still_frames = 0
            
            # Verificar en base a la distancia si el Índice y el pulgar se están casi tocando
            d_pinch = dist(index_pos, thumb_pos)
            is_pinching = d_pinch < PINCH_THRESHOLD
            
            if is_pinching:
                # ESTÁ HACIENDO PINZA (INTENCIÓN: JALAR PARA CONECTAR DOS NODOS)
                if not pinch_active:
                    pinch_active = True
                    # Justo acaba de cerrar los dedos. Buscar si los cerró sobre un nodo existente.
                    closest_id = get_closest_node(index_pos, HOLD_RADIUS * 1.5)
                    if closest_id is not None:
                        dragging_from_node_id = closest_id # Archivar id nodo origen del que tiramos el hilo
                else:
                    # Sigue manteniendo la pinza viva (está arrastrando la mano)
                    if dragging_from_node_id is not None:
                        start_node = get_node_by_id(dragging_from_node_id)
                        if start_node:
                            # Trazar linea/hilo amarillo brillante falso que le sigue mostrando intención de enlace a pantalla
                            cv2.line(frame, (start_node['x'], start_node['y']), index_pos, (0, 255, 255), 2)
                
                # Si está conectando líneas no queremos que accidentalmente se creen nodos de fondo
                index_still_frames = 0
                last_index_pos = None
            else:
                # ACABA DE SOLTAR LA PINZA
                if pinch_active:
                    pinch_active = False
                    if dragging_from_node_id is not None:
                        # Buscamos a qué nodo de destino la acaba de soltar su pinza
                        closest_id = get_closest_node(index_pos, HOLD_RADIUS * 1.5)
                        # Verificamos que sea un nodo destino diferente a sí mismo para ignorarlo
                        if closest_id is not None and closest_id != dragging_from_node_id:
                            # Lo inyectamos usando tuple+sorted para evitar duplicados invertidos (Ej 1-2 vs 2-1)
                            edge = tuple(sorted([dragging_from_node_id, closest_id]))
                            edges.add(edge) # Inyectado permanentemente al set de conexiones graph
                        dragging_from_node_id = None
                
                # SIGUE SOSTENIENDO EL DEDO ÍNDICE SIN PINZA (INTENCIÓN DE DEJAR UN NODO)
                if last_index_pos is None:
                    last_index_pos = index_pos
                    index_still_frames = 1
                elif dist(index_pos, last_index_pos) < HOLD_RADIUS:
                    # El dedo continúa tolerablemente en el mismo cajón de cercanía a la vez anterior
                    index_still_frames += 1
                    
                    # Cargar un círculo verde de feedback simulando progreso a los 3 segundos de meta
                    progress = min(1.0, index_still_frames / FRAMES_TO_CREATE)
                    cv2.circle(frame, last_index_pos, int(NODE_RADIUS * progress), (0, 255, 0), 3)
                    
                    if index_still_frames >= FRAMES_TO_CREATE:
                        # ¡Llenó el anillo temporal! Evitar re-crearse si ya había uno pegado ahí en la misma posición.
                        if get_closest_node(last_index_pos, NODE_RADIUS * 2) is None:
                            # Inyectado permanentemente a la lista estructural del Proyecto-Grafo en memoria.
                            nodes.append({'id': next_node_id, 'x': last_index_pos[0], 'y': last_index_pos[1], 'color': (0, 215, 255)})
                            next_node_id += 1
                            message_text = f"Nodo {next_node_id-1} creado"
                            message_frames = 60
                        index_still_frames = 0
                else:
                    # Movió el dedo más allá de la tolerancia aceptada, reiniciar el reloj cronometrado y la posición base original
                    last_index_pos = index_pos
                    index_still_frames = 0
        else:
            # === NO HAY MANOS O SE PERDIÓ LA CONFIRMACIÓN EN EL PERIMETRO CAMARA ===
            # Reseteamos absoltamente todos los estados de iteraciones en curso que mantuvieran cosas vivas.
            last_index_pos = None
            index_still_frames = 0
            pinch_active = False
            dragging_from_node_id = None
            last_fist_pos = None
            fist_still_frames = 0
            smooth_index = None
            smooth_thumb = None
            smooth_center = None

        # ========================================================
        # 8. SISTEMA DE RENDERIZACIÓN DEFINITIVA A PANTALLA
        # ========================================================
        
        # Iterar sobre las aristas e inyectar líneas en pantalla como paso 1, color blanco
        for edge in edges:
            n1 = get_node_by_id(edge[0])
            n2 = get_node_by_id(edge[1])
            if n1 and n2:
                cv2.line(frame, (n1['x'], n1['y']), (n2['x'], n2['y']), (255, 255, 255), 3)

        # Iterar sobre los nodos e inyectar sobre el diseño una capa gruesa de color y siluetas para enmarcar con circulos
        for node in nodes:
            cv2.circle(frame, (node['x'], node['y']), NODE_RADIUS, node['color'], -1)
            cv2.circle(frame, (node['x'], node['y']), NODE_RADIUS, (0, 0, 0), 2)
            # Renderizar el texto (ID número) centrándolo matemáticamente dentro del nodo restando el alto/ancho de la fuente local
            text_size = cv2.getTextSize(str(node['id']), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            txt_x = node['x'] - text_size[0] // 2
            txt_y = node['y'] + text_size[1] // 2
            cv2.putText(frame, str(node['id']), (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # ========================================================
        # 9. INTERFAZ DE USUARIO TEXTUAL Y ENTRADAS FÍSICAS AL TECLADO (I/O)
        # ========================================================
        cv2.putText(frame, "Cerrar:'Q' | Guardar:'S' | Crear Nodo: Indice 3s | Borrar: Puno 3s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # Renderizador temporal del cartel estelar (Ej. "Guardado como PNG", "Arista Eliminada")
        if message_frames > 0:
            cv2.putText(frame, message_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            message_frames -= 1 # Se descuenta uno a la vez hasta que regrese al infinito

        cv2.imshow('Grafo AR', frame) # Mostrar ahora sí la pantalla al cliente

        # Control C / Kernel local del teclado OS
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Si apretas la q se sale del while destrozándolo y escapando al exterior
            break
        elif key == ord('s'): 
            try:
                # Tomar la copia de matriz de píxeles actual (incluidos los dibujos superpuestos Media) a un archivo .PNG de graficos guardados estáticos.
                cv2.imwrite('grafo.png', frame)
                message_text = "Guardado exitosamente como grafo.png"
                message_frames = 60
                print("Grafo guardado exitosamente en tu explorador de archivos.")
            except Exception as e:
                print(f"Error al guardar de Windows: {e}")

    # Purgar cache memory / quitamos control de kernel a la cámara para devolverlo a ti, al cliente
    cap.release()
    cv2.destroyAllWindows()

# Boilerplate en root. Archivo inicial
if __name__ == '__main__':
    main()