#1. OBSERVAMOS LAS DIMENSIONES DE LOS DATOS
import h5py as h5
import numpy as np
import time
import pandas as pd
import os
import json
import pdb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader 
import os
import gc  # <--- Importamos el recolector de basura
import matplotlib.pyplot as plt



# 1. Configuración de la ruta
ruta_maestra = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/process_data.h5"

ruta_carpeta_guardar= r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/runtransformermulticanal3"
# Crear la carpeta si no existe
os.makedirs(ruta_carpeta_guardar, exist_ok=True)

with h5.File(ruta_maestra, 'r') as f:
    # Leemos las dimensiones de las tres componentes
    grupo = f['data_analisis']
    u_shape = grupo['u_fluc'].shape
    v_shape = grupo['v_fluc'].shape
    w_shape = grupo['w_fluc'].shape

# Verificación de seguridad: las 3 matrices deben ser idénticas en tamaño
assert u_shape == v_shape == w_shape, "¡Error crítico! Las dimensiones de U, V y W no coinciden."

num_tiempos = u_shape[0]
dim_y = u_shape[1] #number of lines of probes
dim_z = u_shape[2]
total_puntos_espaciales = dim_y * dim_z

print("--- ESTADÍSTICAS DEL DATASET MULTICANAL ---")
print(f"Total de instantes de tiempo: {num_tiempos}")
print(f"Malla espacial: {dim_y} x {dim_z} = {total_puntos_espaciales} puntos")
print(f"Canales de velocidad: 3 (U, V, W)")
print(f"Total de datos numéricos a procesar: {num_tiempos * total_puntos_espaciales * 3:,}")

# 2. CÁLCULO DE DIVISIONES (Splits temporales)
# Primero separamos el 20% para el examen final (test/resultados)
indice_test = int(num_tiempos * 0.8)

# Del 80% restante, separamos otro 20% para validación (0.8 * 0.8 = 0.64)
indice_val = int(indice_test * 0.8)

salto = 50

print("\n--- REPARTO DEL TIEMPO ---")
print(f"1. Entrenamiento (64%): Pasos 0 al {indice_val}")
print(f"2. Validación    (16%): Pasos {indice_val} al {indice_test}")
print(f"3. Resultados    (20%): Pasos {indice_test} al {num_tiempos}")







#2. PREPARACIÓN DE LOS DATOS PARA EL MODELO

# 2. PREPARACIÓN DE LOS DATOS PARA EL MODELO

def obtener_o_crear_datasets(ruta_maestra, train_dataset_pt_tf, val_dataset_pt_tf, stats_pt_tf):
    if os.path.exists(train_dataset_pt_tf) and os.path.exists(val_dataset_pt_tf) and os.path.exists(stats_pt_tf):
        print(f"📦 Archivos detectados. Cargando datasets desde {train_dataset_pt_tf}...")
        train_ds = torch.load(train_dataset_pt_tf, weights_only=False)
        val_ds = torch.load(val_dataset_pt_tf, weights_only=False)
        stats = torch.load(stats_pt_tf, weights_only=False)
        return train_ds, val_ds, stats

    else:
        with h5.File(ruta_maestra, 'r') as f:
            # 🚨 1. EXTRAEMOS LAS 7 LÍNEAS DE GOLPE 
            # Forma: (106000 tiempos, 7 líneas, 288 puntos Z)
            matriz_base = f['data_analisis']['u_fluc'][:, :, :] 

        matriz_plana = matriz_base.reshape(matriz_base.shape[0], -1) 
        
        pasos_maximos = 3000
        dim_total = matriz_plana.shape[1]
        
        mapa_coherencia_u = np.full(dim_total, pasos_maximos, dtype=int)
        ya_encontrado_u = np.zeros(dim_total, dtype=bool)
        
        for k in range(1, pasos_maximos):
            covarianza_u = np.sum(matriz_plana[:-k, :] * matriz_plana[k:, :], axis=0)
            cruzaron_u = (covarianza_u < 0) & (~ya_encontrado_u)
            mapa_coherencia_u[cruzaron_u] = k
            ya_encontrado_u[cruzaron_u] = True
            if np.all(ya_encontrado_u): break

        # Calculamos la media de TODOS los puntos del espacio 2D
        if np.any(ya_encontrado_u):
            media_autocorrelacion = int(np.mean(mapa_coherencia_u[ya_encontrado_u]))
        else:
            media_autocorrelacion = pasos_maximos
            
        print(f"Longitud media de autocorrelación GLOBAL: {media_autocorrelacion} time steps")

        # 2. Dividimos en chunks el cubo 3D
        chunk_size = 550 
        num_chunks = len(matriz_base) // chunk_size 
        matriz_recortada = matriz_base[:num_chunks * chunk_size, :, :]

        trozos = []
        for i in range(num_chunks):
            trozos.append(matriz_recortada[i*chunk_size:(i+1)*chunk_size, :, :])

        # Concatenamos a lo largo del eje Z (eje 2) para tener miles de ejemplos espaciales
        matriz_final_raw = np.concatenate(trozos, axis=2) # Forma: (5050, 7, 5760)

        # 5. DIVISIÓN 80/20 DE LOS PUNTOS ESPACIALES (5760)
        total_puntos = matriz_final_raw.shape[2] 
        split_idx = int(total_puntos * 0.8) 

        np.random.seed(42) 
        indices_barajados = np.random.permutation(total_puntos)
        train_indices = indices_barajados[:split_idx] 
        val_indices = indices_barajados[split_idx:]   

        matriz_train_raw = matriz_final_raw[:, :, train_indices] # (5050, 7, 4608)
        matriz_val_raw = matriz_final_raw[:, :, val_indices]     # (5050, 7, 1152)

        # Normalización global
        media_X = np.mean(matriz_train_raw)
        std_X = np.std(matriz_train_raw)
        stats = {'media': float(media_X), 'std': float(std_X)}

        matriz_train = np.float32((matriz_train_raw - media_X) / std_X)
        matriz_val = np.float32((matriz_val_raw - media_X) / std_X)

        del matriz_final_raw, matriz_train_raw, matriz_val_raw, trozos, matriz_recortada
        gc.collect()

        # ==========================================
        # 6. CREACIÓN DE VENTANAS (LA SEPARACIÓN 6/1)
        # ==========================================
        seq_x = int(media_autocorrelacion * 2) 
        seq_y = 50   
        salto = 50   

        # 🚨 NUEVA FUNCIÓN EXTRACTORA: INVARIANZA ESPACIAL 🚨
    def crear_ventanas_universales_cfd(matriz, lookback, predict, stride):
        X_lista, y_lista = [], []
        num_tiempos = matriz.shape[0]
    
        print("Generando dataset universal (deslizamiento espacio-temporal)...")
    # Bucle 1: Deslizamiento Temporal (Avanzamos en el tiempo)
        for i in range(0, num_tiempos - lookback - predict + 1, stride):
        
        # Bucle 2: Deslizamiento Espacial (¡La magia del Camino B!)
        # Tenemos 7 líneas en total (0 a 6). 
        # offset 0: X = [0,1,2,3,4] -> Y = [5]  (Aprende la zona de pared)
        # offset 1: X = [1,2,3,4,5] -> Y = [6]  (Aprende la zona exterior)
            for offset in [0, 1]: 
            # Recortamos 5 líneas de entrada
                bloque_x = matriz[i : i + lookback, offset : offset + 5, :]
            
            # Recortamos la 6ª línea como objetivo
                bloque_y = matriz[i + lookback : i + lookback + predict, offset + 5 : offset + 6, :]
            
            # Transponemos a (N_pts, seq_len, canales) y guardamos
                X_lista.append(bloque_x.transpose(2, 0, 1))
                y_lista.append(bloque_y.transpose(2, 0, 1))
            
        return np.vstack(X_lista), np.vstack(y_lista)

    print("\nGenerando ventanas CFD (X=5 líneas, Y=1 línea objetivo)...")
    X_train_np_tf, y_train_np_tf = crear_ventanas_universales_cfd(matriz_train, seq_x, seq_y, salto)
    X_val_np_tf, y_val_np_tf = crear_ventanas_universales_cfd(matriz_val, seq_x, seq_y, salto)

    train_features = torch.from_numpy(X_train_np_tf)
    val_features = torch.from_numpy(X_val_np_tf)
    train_labels = torch.from_numpy(y_train_np_tf)
    val_labels = torch.from_numpy(y_val_np_tf)

    del X_train_np_tf, y_train_np_tf, X_val_np_tf, y_val_np_tf
    gc.collect()

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    torch.save(train_dataset, train_dataset_pt_tf)
    torch.save(val_dataset, val_dataset_pt_tf)
    torch.save(stats, stats_pt_tf)

    return train_dataset, val_dataset, stats


# --- MODO DE USO ---
# Define los nombres de tus archivos
f_train = 'train_dataset_tfmulticanal3.pt'
f_val = 'val_dataset_tfmulticanal3.pt'
f_stats = 'norm_stats_tfmulticanal3.pt'

# Llamas a la función
train_dataset, val_dataset, stats_norm = obtener_o_crear_datasets(ruta_maestra, f_train, f_val, f_stats)

train_loader= DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=False, num_workers=4, pin_memory=True)
val_loader= DataLoader(val_dataset, batch_size=64, shuffle=False,drop_last=False, num_workers=4, pin_memory=True)


#3. DEFINIMOS NUESTRO MODELO TRANSFORMER



class MultiHeadEasyAttention(nn.Module):
    def __init__(self, seq_len, input_dim, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Tensor \alpha para múltiples cabezas: (h, p, p)
        self.alpha = nn.Parameter(torch.empty(num_heads, seq_len, seq_len))
        nn.init.xavier_uniform_(self.alpha)
        
        # Proyección compartida W_V para todas las cabezas
        self.W_v = nn.Linear(input_dim, d_model, bias=False)

    def forward(self, x):
        b, p, _ = x.size() # b=Batch, p=seq_len
        
        # 1. Proyectar valores: (Batch, seq_len, d_model)
        V = self.W_v(x)
        
        # 2. Separar en múltiples cabezas: (Batch, seq_len, num_heads, head_dim)
        V = V.view(b, p, self.num_heads, self.head_dim)
        # Transponer para emparejar con alpha: (Batch, num_heads, seq_len, head_dim)
        V = V.permute(0, 2, 1, 3)
        
        # 3. Multiplicar cada cabeza de \alpha con su bloque V correspondiente
        # 'hij' (alpha), 'bhjd' (V) -> 'bhid' (salida por cabeza)
        out = torch.einsum('hij,bhjd->bhid', self.alpha, V)
        
        # 4. Concatenar las cabezas de vuelta al tamaño d_model
        out = out.permute(0, 2, 1, 3).contiguous().view(b, p, -1)
        return out
    

class EasyTransformerBlock(nn.Module):
    # 🚨 NUEVO: DROPOUT 🚨 Se añade como parámetro (0.1 = 10% de neuronas apagadas)
    def __init__(self, seq_len, d_model=64, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadEasyAttention(seq_len, d_model, d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 🚨 NUEVO: DROPOUT 🚨 Después de la atención
        self.dropout1 = nn.Dropout(dropout_rate)
        
        expansion_factor = 4
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor), 
            nn.ReLU(),
            # 🚨 NUEVO: DROPOUT 🚨 Dentro del bloque Feed Forward
            nn.Dropout(dropout_rate),
            nn.Linear(d_model * expansion_factor, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # 🚨 NUEVO: DROPOUT 🚨 Después del Feed Forward
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Bloque 1: Atención + Dropout + Residual + Normalización
        attn_out = self.attention(x)
        attn_out = self.dropout1(attn_out) # 🚨 Aplicamos Dropout 1
        x = self.norm1(x + attn_out)
        
        # Bloque 2: Feed Forward + Dropout + Residual + Normalización
        ff_out = self.feed_forward(x)
        ff_out = self.dropout2(ff_out)     # 🚨 Aplicamos Dropout 2
        x = self.norm2(x + ff_out)
        return x

class EasyFluidPredictor(nn.Module):
    # 🚨 NUEVO PARÁMETRO: num_layers (Por defecto 2)
    def __init__(self, seq_len=500, pred_len=50, input_dim=5, d_model=128, num_heads=8, dropout_rate=0.2, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1 
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # 🚨 LA MAGIA DEL NIVEL 2: Apilamos múltiples bloques Transformer 🚨
        # nn.ModuleList crea una lista de bloques independientes. Cada uno tendrá sus propios pesos.
        self.transformer_blocks = nn.ModuleList([
            EasyTransformerBlock(seq_len, d_model, num_heads, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        self.temporal_proj = nn.Linear(seq_len, pred_len)  
        self.final_proj = nn.Linear(d_model, self.output_dim)    

    def forward(self, x):
        ultimo_valor_adyacente = x[:, -1:, -1:] 

        x_trans = self.input_proj(x)            
        x_trans = self.input_dropout(x_trans)   
        
        # 🚨 Pasamos la información a través de TODAS las capas apiladas secuencialmente
        for block in self.transformer_blocks:
            x_trans = block(x_trans)
        
        x_trans = x_trans.permute(0, 2, 1)      
        x_trans = self.temporal_proj(x_trans)   
        x_trans = x_trans.permute(0, 2, 1)      
        
        delta_u = self.final_proj(x_trans)      
        
        out = ultimo_valor_adyacente + delta_u
        return out
    

#4. ENTRENAMOS NUESTRA RED NEURONAL
# ==========================================
# 1. CONFIGURACIÓN DEL HARDWARE
# ==========================================

# Detectar si hay Tarjeta Gráfica (GPU) disponible. Si no, usará la CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en el dispositivo: {device}")


# 🚨 AUTODETECCIÓN DE DIMENSIONES 🚨
# train_dataset[0] devuelve la primera tupla (X, Y)
ejemplo_x, ejemplo_y = train_dataset[0]

num_heads = 8  # Puedes ajustar este número según la capacidad de tu GPU y la complejidad que quieras

d_model = 128  # Dimensión interna del modelo (puedes ajustar este número para hacerlo más pequeño o más grande)

num_layers = 2  # Número de bloques Transformer apilados (puedes probar con 1, 2 o 3)

input_dim = train_dataset.tensors[0].shape[2]  # Cada punto espacial es un canal de entrada

pred_len = train_dataset.tensors[1].shape[1]  # Número de pasos futuros a predecir

seq_len = train_dataset.tensors[0].shape[1]  # Número de pasos pasados que el modelo usará para predecir

lr_maximo=0.001  # Tasa de aprendizaje para el optimizador

dropout_rate=0.2  # Tasa de Dropout

epocas = 500    # Número de veces que la IA verá TODO el dataset completo



modelo = EasyFluidPredictor(seq_len=seq_len, pred_len=pred_len, input_dim=input_dim, d_model=d_model, num_heads=num_heads,dropout_rate=dropout_rate, num_layers=num_layers)

# Movemos el modelo a la tarjeta gráfica (o lo dejamos en CPU)
modelo.to(device)


criterio = torch.nn.MSELoss()

# --- 🚀 NUEVO: CONFIGURACIÓN DE PÉRDIDA POR GRADIENTE ---
# 0.1 o 0.2 suele ser el "punto dulce" para empezar. 
# Si notas que la línea roja se vuelve demasiado caótica o "nerviosa", baja este valor a 0.05.
peso_gradiente = 0.1

# --- 🚀 NUEVO: SCHEDULER DE TASA DE APRENDIZAJE ---
# Si la validación (val_loss) no mejora durante 15 épocas, bajamos el learning rate a la mitad (0.001)
# 🚨 FUERA ReduceLROnPlateau, ENTRA OneCycleLR 🚨
optimizador = torch.optim.Adam(modelo.parameters(), lr=lr_maximo, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizador,
    max_lr=lr_maximo,
    epochs=epocas,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # Dedica el 10% inicial del entrenamiento solo a calentar
    anneal_strategy='cos' # Baja el LR formando una curva suave (coseno)
)


# Acelerador de hardware de PyTorch
scaler = torch.cuda.amp.GradScaler()



# ==========================================
# 2. PARÁMETROS DEL ENTRENAMIENTO
# ==========================================

tek=time.time()


historial_train_loss = []
historial_val_loss = []
#Creamos para guadar el mejor modelo basado en la pérdida de validación
best_val_loss = float('inf')  # Empezamos con "infinito" para que cualquier pérdida sea menor
ruta_mejor_modelo = ruta_carpeta_guardar+'/mejor_modelo_ia_transformer_eas.pt'
ruta_ultimo_modelo = ruta_carpeta_guardar+'/ultimo_modelo_ia_transformer_eas.pt'

print("\n🚀 ¡Iniciando el entrenamiento de la Red Neuronal!")
print("-" * 60)

# ==========================================
# 3. EL BUCLE PRINCIPAL
# ==========================================
for epoca in range(epocas):
    print(f"\n📊 Época {epoca+1}/{epocas}")
    tic = time.time()
    
    # --- FASE DE ENTRENAMIENTO ---
# --- FASE DE ENTRENAMIENTO ---
    modelo.train()  
    train_loss_acumulada = 0.0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizador.zero_grad()
        
        with torch.cuda.amp.autocast():
            predicciones = modelo(x_batch)
            loss_base = criterio(predicciones, y_batch)
            diff_pred = predicciones[:, 1:, :] - predicciones[:, :-1, :]
            diff_real = y_batch[:, 1:, :] - y_batch[:, :-1, :]
            loss_gradient = torch.mean(torch.abs(diff_pred - diff_real)) 
            loss = loss_base + (peso_gradiente * loss_gradient)
        
        scaler.scale(loss).backward()
        scaler.step(optimizador)
        scaler.update()
        
        # 🚨 NUEVO: El scheduler actualiza el LR en cada paso (lote)
        scheduler.step() 
        
        train_loss_acumulada += loss.item()

        
    # Calculamos el error medio de esta época
    avg_train_loss = train_loss_acumulada / len(train_loader)
    historial_train_loss.append(avg_train_loss)
    
    
   # --- FASE DE VALIDACIÓN ---
    modelo.eval()  # Modo examen
    val_loss_acumulada = 0.0
    
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            # 🚀 Añadimos el autocast aquí también para que valide a la velocidad de la luz
            with torch.cuda.amp.autocast():
                predicciones_val = modelo(x_val)
                
                # --- VALIDACIÓN CON LA MISMA MÉTRICA ---
                loss_base_val = criterio(predicciones_val, y_val)
                diff_pred_val = predicciones_val[:, 1:, :] - predicciones_val[:, :-1, :]
                diff_real_val = y_val[:, 1:, :] - y_val[:, :-1, :]
                loss_gradient_val = torch.mean(torch.abs(diff_pred_val - diff_real_val))
                
                loss_val = loss_base_val + (peso_gradiente * loss_gradient_val)
                
            # IMPORTANTE: Sacamos el .item() fuera del bloque autocast
            val_loss_acumulada += loss_val.item()
            
    # Calculamos el error medio del examen
    avg_val_loss = val_loss_acumulada / len(val_loader)
    historial_val_loss.append(avg_val_loss)

 # Guardamos el Learning Rate ANTES de que actúe el scheduler
    lr_anterior = optimizador.param_groups[0]['lr']
    
    # Le pasamos la nota de validación al Scheduler para que decida si baja el LR
    scheduler.step(avg_val_loss)
    
    # Miramos el Learning Rate DESPUÉS
    lr_nuevo = optimizador.param_groups[0]['lr']
    
    if lr_nuevo < lr_anterior:
        print(f"\n📉 ¡ATENCIÓN! El modelo se había estancado. Learning Rate reducido de {lr_anterior} a {lr_nuevo}\n")

    toc = time.time()

    # --- GUARDADO DEL MEJOR MODELO ---
    if avg_val_loss < best_val_loss:
        print(f"⭐ ¡Nuevo récord! La pérdida bajó de {best_val_loss:.6f} a {avg_val_loss:.6f}. Guardando...")
        
        best_val_loss = avg_val_loss
        
        checkpoint = {
            'epoch': epoca,
            'model_state_dict': modelo.state_dict(),
            'optimizer_state_dict': optimizador.state_dict(),
            'loss': best_val_loss,
            'stats': stats_norm 
        }
        torch.save(checkpoint, ruta_mejor_modelo)

    # Imprimimos el progreso al final de cada época
    print(f"Época [{epoca+1:02d}/{epocas}] | Train Loss (Compuesta): {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    print(f"Tiempo de la época: {toc - tic:.2f} segundos")

print("-" * 60)
print("✅ ¡Entrenamiento completado!")
tac=time.time()

print('Guardando último modelo entrenado (sin importar si es el mejor o no)...')
checkpoint = {
    'epoch': epoca,
    'model_state_dict': modelo.state_dict(),
    'optimizer_state_dict': optimizador.state_dict(),
    'loss': avg_val_loss,
    'stats': stats_norm 
}
torch.save(checkpoint, ruta_ultimo_modelo)


# ==========================================
# 5. GUARDADO DE METADATOS Y RESULTADOS (JSON)
# ==========================================


# Creamos el diccionario adaptado a la arquitectura Transformer
metadata = {}

# 1. Arquitectura del Transformer (Los "músculos" del modelo)
metadata['model_type'] = "Easy Attention Transformer"
metadata['input_dim'] = input_dim         # Dimensión de entrada (velocidad u)
metadata['seq_len'] = seq_len             # Ventana de pasado (lookback)
metadata['pred_len'] = pred_len             # Ventana de futuro (prediction)
metadata['d_model'] = d_model              # Espacio latente interno
metadata['num_heads'] = num_heads             # Número de cabezas de atención

# 2. Hiperparámetros de entrenamiento
metadata['learning_rate'] = f"{lr_maximo} (Va decreciendo)"   # El LR que pusimos en el Adam
metadata['epochs_planned'] = epocas   # Total de épocas solicitadas
metadata['peso_gradiente'] = peso_gradiente  # 🚨 NUEVO: Peso de la función de pérdida física
metadata['dropout_rate'] = dropout_rate              # 🚨 NUEVO: Tasa de Dropout usada (cámbialo si usaste otro)
metadata['num_layers'] = num_layers              # 🚨 NUEVO: Número de bloques Transformer apilados        
# 3. Estadísticas de rendimiento (El "marcador")
metadata['best_val_loss'] = best_val_loss
metadata['salto_ventana_deslizante'] = salto
metadata['train_time_seconds'] = tac - tek
metadata['train_time_minutes'] = (tac - tek) / 60
metadata['date_time'] = time.strftime("%Y-%m-%d %H:%M:%S")

# 4. Información del Dataset y Hardware
metadata['num_train_samples'] = len(train_loader.dataset)
metadata['num_val_samples'] = len(val_loader.dataset)
metadata['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"

# 5. Estadísticas de Normalización (¡Crucial para usar el modelo después!)
# Guardamos media y std para poder des-normalizar predicciones sin cargar el .pt
metadata['norm_stats'] = {
    'media': float(stats_norm['media']),
    'std': float(stats_norm['std'])
}

# Guardar el diccionario en un archivo JSON bien formateado
ruta_json = ruta_carpeta_guardar + '/hyperparams_transformer.json'

with open(ruta_json, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)

print(f"✅ Metadatos guardados correctamente en: {ruta_json}")
#pdb.set_trace()
# 6. VISUALIZACIÓN DE RESULTADOS

'''
print("Calculando el Baseline Ingenuo para la gráfica...")

# ==========================================
# 1. CÁLCULO DEL BASELINE (La predicción "estúpida")
# ==========================================
val_loss_ingenua_acumulada = 0.0
criterio_baseline = nn.MSELoss()

with torch.no_grad():
    for x_val, y_val in val_loader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        
        # 1. Cogemos el último paso de la ventana de entrada (forma: [64])
        ultimo_valor = x_val[:, -1] 
        
        # 2. Le añadimos una dimensión y lo repetimos 50 veces para igualar al futuro
        # Forma final: [64, 50]
        prediccion_estupida = ultimo_valor.unsqueeze(1).repeat(1, y_val.shape[1])
        
        # 3. Ahora sí podemos comparar 50 predicciones idénticas con 50 realidades
        loss_estupida = criterio_baseline(prediccion_estupida, y_val)
        val_loss_ingenua_acumulada += loss_estupida.item()

baseline_val_loss = val_loss_ingenua_acumulada / len(val_loader)
print(f"🚨 Valor del Baseline (MSE): {baseline_val_loss:.6f}")
'''
# ==========================================
# 2. DIBUJAMOS LA GRÁFICA
# ==========================================
plt.figure(1,figsize=(10, 6))
plt.clf()
# Eje X: Número de épocas (de 1 a N)
epocas_x = range(1, len(historial_train_loss) + 1)

#Convertimos en arrays
historial_val_loss_np = np.array(historial_val_loss)
epocas_x_np = np.array(epocas_x)

# Trazamos las curvas de la IA
plt.semilogy(epocas_x, historial_train_loss, 'b-o', label='Train Loss (Entrenamiento)', linewidth=2)
plt.semilogy(epocas_x, historial_val_loss, 'g-s', label='Val Loss (Validación)', linewidth=2)
plt.semilogy(epocas_x[historial_val_loss_np.argmin()], historial_val_loss_np.min(), '*r', label='Best', markersize=15)

# Trazamos la línea infranqueable del baseline
#plt.axhline(y=baseline_val_loss, color='r', linestyle='--', linewidth=2, label='Baseline (Persistencia)')

# Estética y etiquetas
plt.title('Curvas de Aprendizaje de la IA en Turbulencia', fontsize=16, fontweight='bold')
plt.xlabel('Épocas', fontsize=14)
plt.ylabel('Pérdida / Error (MSE)', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
#plt.xticks(epocas_x) # Fuerza a que el eje X muestre números enteros

# Ajustamos márgenes y mostramos
plt.tight_layout()
plt.savefig(ruta_carpeta_guardar+'/curvas_aprendizaje.png', dpi=300) # Guardamos la figura en alta resolución
#plt.show()


df = pd.DataFrame()
df['epoch']=range(1, len(historial_train_loss)+1)
df['train_loss']=historial_train_loss
df['val_loss']=historial_val_loss
df.to_csv(ruta_carpeta_guardar+'/historial_perdidas.csv', index=False, sep=',', float_format='%.6e')


