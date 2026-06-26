#1. OBSERVAMOS LAS DIMENSIONES DE LOS DATOS
import h5py as h5
import numpy as np
import time
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader 
# 1. Configuración de la ruta
ruta_maestra = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/process_data.h5"

nombre_carpeta = input("Ingresa el nombre de la carpeta para guardar los resultados (se creará si no existe): ")

# 2. Unes la ruta base con el nombre de la carpeta de forma segura
ruta_carpeta_guardar = os.path.join(r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados", nombre_carpeta)


# Crear la carpeta si no existe
os.makedirs(ruta_carpeta_guardar, exist_ok=True)

def calcular_autocorrelacion(ruta_maestra, linea_idy, pasos_maximos=3000):
    """
    Calcula la longitud media de autocorrelación para una línea específica del archivo HDF5.
    """
    print(f"🔍 Calculando autocorrelación para la línea {linea_idy}...")
    
    # Leer solo la matriz necesaria para la línea seleccionada
    with h5.File(ruta_maestra, 'r') as f:
        matriz_plana = f['data_analisis']['u_fluc'][:, linea_idy, :]  # Forma: (T, N_spatial)
        
    dim_total = matriz_plana.shape[1]
    mapa_coherencia_u = np.full(dim_total, pasos_maximos, dtype=int)
    ya_encontrado_u = np.zeros(dim_total, dtype=bool)
            
    for k in range(1, pasos_maximos):
        covarianza_u = np.sum(matriz_plana[:-k, :] * matriz_plana[k:, :], axis=0)
        cruzaron_u = (covarianza_u < 0) & (~ya_encontrado_u)
        mapa_coherencia_u[cruzaron_u] = k
        ya_encontrado_u[cruzaron_u] = True
        if np.all(ya_encontrado_u): 
            break
    # Calculamos la media de TODOS los puntos
    if np.any(ya_encontrado_u):
        media_autocorrelacion = int(np.mean(mapa_coherencia_u[ya_encontrado_u]))
    else:
        media_autocorrelacion = pasos_maximos
                
    print(f"✅ Longitud media de autocorrelación (Línea {linea_idy}): {media_autocorrelacion} time steps\n")
    return media_autocorrelacion



#2. PREPARACIÓN DE LOS DATOS PARA EL MODELO

def obtener_o_crear_datasets(ruta_maestra, train_dataset_pt, val_dataset_pt, stats_pt, linea_idy,
                              window_size=275, step=25, input_limit=250):
    """
    Parámetros nuevos:
    - window_size : tamaño total de la ventana (input + output). Default 275.
    - step        : desplazamiento entre ventanas consecutivas. Default 25.
                    step = window_size → comportamiento original (sin solapamiento).
    - input_limit : timesteps de entrada. Output = window_size - input_limit = 25.
    """

    if os.path.exists(train_dataset_pt) and os.path.exists(val_dataset_pt) and os.path.exists(stats_pt):
        print(f"📦 Archivos detectados. Cargando datasets...")
        train_ds = torch.load(train_dataset_pt, weights_only=False)
        val_ds   = torch.load(val_dataset_pt,   weights_only=False)
        stats    = torch.load(stats_pt,          weights_only=False)
        return train_ds, val_ds, stats

    # --- LECTURA ---
    with h5.File(ruta_maestra, 'r') as f:
        matriz_base = f['data_analisis']['u_fluc'][:, linea_idy, :]  # (T, 288)

    T, N_spatial = matriz_base.shape

    # --- VENTANAS DESLIZANTES ---
    trozos = []
    for start in range(0, T - window_size + 1, step):
        trozos.append(matriz_base[start : start + window_size, :])  # (window_size, 288)

    n_ventanas = len(trozos)
    print(f"  T={T} | window_size={window_size} | step={step}")
    print(f"  Ventanas: {n_ventanas}  (vs {T // window_size} sin solapamiento → ×{n_ventanas // (T // window_size):.0f} más datos)")

    matriz_final = np.hstack(trozos)  # (window_size, n_ventanas * N_spatial)
    print(f"  Forma de matriz_final: {matriz_final.shape}")

    # --- SEPARAR X / y ---
    X_data_raw = matriz_final[:input_limit, :]  # (250, N_cols)
    y_data_raw = matriz_final[input_limit:, :]  # (25,  N_cols)
    print(f"  X_data_raw: {X_data_raw.shape}  |  y_data_raw: {y_data_raw.shape}")

    # --- SPLIT TEMPORAL 80/20 ---
    # ⚠️ Con solapamiento, el split aleatorio por columna genera fuga de datos:
    # la ventana i y la i+1 comparten el 90% de timesteps. Usar split temporal
    # garantiza que train y val no comparten ningún timestep.
    split_ventana = int(n_ventanas * 0.8)
    train_cols = split_ventana * N_spatial

    X_train = X_data_raw[:, :train_cols]
    X_val   = X_data_raw[:, train_cols:]
    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}")

        # --- NORMALIZACIÓN ---
    media_X = np.mean(X_train)
    std_X   = np.std(X_train)
    stats = {'media': media_X, 'std': std_X}

    X_data = (X_data_raw - media_X) / std_X
    y_data = (y_data_raw - media_X) / std_X

    X_train_np = X_data[:, :train_cols]
    X_val_np   = X_data[:, train_cols:]
    y_train_np = y_data[:, :train_cols]
    y_val_np   = y_data[:, train_cols:]

    # --- TENSORES y DATASETS ---
    train_dataset = TensorDataset(
        torch.tensor(X_train_np.T, dtype=torch.float32),
        torch.tensor(y_train_np.T, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_np.T,   dtype=torch.float32),
        torch.tensor(y_val_np.T,   dtype=torch.float32)
    )

    torch.save(train_dataset, train_dataset_pt)
    torch.save(val_dataset,   val_dataset_pt)
    torch.save(stats,         stats_pt)

    return train_dataset, val_dataset, stats

# Seleccionamos la línea o matriz que nos interesa 
linea_idy=2
input_limit=int(calcular_autocorrelacion(ruta_maestra, linea_idy, pasos_maximos=3000)*0.15)  # 30% de la longitud media de autocorrelación
step=int(input_limit * 0.1)  # 10% de input_limit
windows_size=step+input_limit

# --- MODO DE USO ---
# Define los nombres de tus archivos
f_train = 'train_dataset_pro3.pt'
f_val = 'val_dataset_pro3.pt'
f_stats = 'norm_stats_pro3.pt'

# Llamas a la función
train_dataset, val_dataset, stats_norm = obtener_o_crear_datasets(
    ruta_maestra, f_train, f_val, f_stats, linea_idy,
    window_size=windows_size, step=step, input_limit=input_limit
)

train_loader= DataLoader(train_dataset, batch_size=128, shuffle=True,drop_last=False, num_workers=4, pin_memory=True)
val_loader= DataLoader(val_dataset, batch_size=128, shuffle=False,drop_last=False, num_workers=4, pin_memory=True)


#3. CREAMOS NUESTRA RED NEURONAL

# 1. Definimos las dimensiones (Solo canal U)
canales = 1
features_totales_in = train_dataset.tensors[0].shape[1]  
features_totales_out = train_dataset.tensors[1].shape[1]  
lstm_hidden_size = 256
num_lstmlayers = 3
learning_rate = 0.0001


# 2. Construcción de la Red Neuronal
class FluidoLSTM_Seq2Seq(nn.Module):
    def __init__(self, hidden_size=256, num_lstmlayers=3, output_size=25, dropout=0.2):
        super(FluidoLSTM_Seq2Seq, self).__init__()
        self.output_size = output_size
        
        # ENCODER: Procesa la historia pasada (los 250 pasos)
        self.encoder = nn.LSTM(input_size=1, 
                               hidden_size=hidden_size, 
                               num_layers=num_lstmlayers, 
                               batch_first=True, 
                               dropout=dropout)
        
        # DECODER: Genera el futuro paso a paso
        self.decoder = nn.LSTM(input_size=1, 
                               hidden_size=hidden_size, 
                               num_layers=num_lstmlayers, 
                               batch_first=True, 
                               dropout=dropout)
        
        # Capa lineal que convierte la salida del decoder en la velocidad predicha (1 solo valor)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, 250)
        x = x.unsqueeze(-1) # -> (batch_size, 250, 1)
        
        # 1. El Encoder lee el pasado. 
        # No nos importa el 'encoder_out', solo queremos sus memorias finales (hidden, cell)
        _, (hidden, cell) = self.encoder(x)
        
        # 2. El primer input para el Decoder es el ÚLTIMO valor conocido de 'x' (el paso 250)
        decoder_input = x[:, -1, :].unsqueeze(1) # -> (batch_size, 1, 1)
        
        predicciones = []
        
        # 3. Bucle Autoregresivo: Predecimos los 25 pasos en cadena
        for _ in range(self.output_size):
            # El Decoder usa el input actual y las memorias (hidden, cell)
            decoder_out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            
            # Calculamos el valor exacto de la velocidad para ESTE timestep
            pred_step = self.linear(decoder_out[:, 0, :]) # -> (batch_size, 1)
            
            # Lo guardamos en nuestra lista
            predicciones.append(pred_step)
            
            # 🔥 MAGIA: La predicción actual se convierte en el input para predecir el siguiente paso
            decoder_input = pred_step.unsqueeze(1) # -> (batch_size, 1, 1)
            
        # Concatenamos todas las predicciones en un solo tensor: (batch_size, 25)
        out = torch.cat(predicciones, dim=1)
        return out    

# 3. Inicializamos el modelo
modelo = FluidoLSTM_Seq2Seq(hidden_size=lstm_hidden_size,
                    num_lstmlayers=num_lstmlayers,
                    output_size=features_totales_out)

# 4. Definimos el "Castigo" (Pérdida) y el "Optimizador"
def physics_loss(prediccion, real, alfa=0.5):
    # Error estándar (MSE normal)
    mse_base = torch.mean((prediccion - real)**2)
    
    # Error de la primera derivada (diferencia entre pasos temporales consecutivos)
    # Esto obliga a la IA a calcar los picos y valles de la turbulencia
    diff_pred = prediccion[:, 1:] - prediccion[:, :-1]
    diff_real = real[:, 1:] - real[:, :-1]
    mse_derivada = torch.mean((diff_pred - diff_real)**2)
    
    # El alfa decide cuánto peso le das a "copiar bien las curvas"
    return mse_base + alfa * mse_derivada, mse_base


optimizador = torch.optim.Adam(modelo.parameters(), lr=learning_rate)

# Imprimimos el modelo para ver su estructura
print(modelo)



#4. ENTRENAMOS NUESTRA RED NEURONAL
# ==========================================
# 1. CONFIGURACIÓN DEL HARDWARE
# ==========================================
# Detectar si hay Tarjeta Gráfica (GPU) disponible. Si no, usará la CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Entrenando en el dispositivo: {device}")

# Movemos el modelo a la tarjeta gráfica (o lo dejamos en CPU)
modelo.to(device)

# ==========================================
# 2. PARÁMETROS DEL ENTRENAMIENTO
# ==========================================
epocas = 200  # Número de veces que la IA verá TODO el dataset completo
tek=time.time()


historial_train_loss = []
historial_val_loss = []
#Creamos para guadar el mejor modelo basado en la pérdida de validación
best_val_loss = float('inf')  # Empezamos con "infinito" para que cualquier pérdida sea menor
ruta_mejor_modelo = ruta_carpeta_guardar+'/mejor_modelo_ia.pt'

print("\n🚀 ¡Iniciando el entrenamiento de la Red Neuronal!")
print("-" * 60)

# ==========================================
# 3. EL BUCLE PRINCIPAL
# ==========================================
for epoca in range(epocas):
    print(f"\n📊 Época {epoca+1}/{epocas}")
    tic = time.time()
    # --- FASE DE ENTRENAMIENTO ---
    modelo.train()  # Ponemos el modelo en modo "aprender"
    train_loss_acumulada = 0.0
    
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        # Movemos los datos a la GPU (o CPU)
        x_batch = x_batch.to(device)
        #print('input',x_batch.shape)
        y_batch = y_batch.to(device)
        #print('output',y_batch.shape)
        # a) Reiniciamos los gradientes (borramos la memoria de la pasada anterior)
        optimizador.zero_grad()
        
        # b) Pasada hacia adelante (Forward): La IA intenta predecir el futuro
        predicciones = modelo(x_batch)
        
        # c) Calculamos el error (Loss): Comparamos la predicción con la realidad
        loss, mse_basetrain = physics_loss(predicciones, y_batch, alfa=0.5)
        
        # d) Pasada hacia atrás (Backward): Calculamos cómo corregir los errores
        loss.backward()

        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        
        # e) Actualizamos los pesos (El aprendizaje real ocurre aquí)
        optimizador.step()
        
        train_loss_acumulada += mse_basetrain.item()

        
        
    # Calculamos el error medio de esta época
    avg_train_loss = train_loss_acumulada / len(train_loader)
    historial_train_loss.append(avg_train_loss)
    
    
    # --- FASE DE VALIDACIÓN ---
    # Aquí la IA NO aprende, solo hace un "examen" con datos que no ha visto
    modelo.eval()  # Ponemos el modelo en modo "examen"
    val_loss_acumulada = 0.0
    
    # Apagamos el cálculo de gradientes para ahorrar memoria y no aprender sin querer
    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            
            predicciones_val = modelo(x_val)
            loss_val, mse_baseval = physics_loss(predicciones_val, y_val, alfa=0.5)
            val_loss_acumulada += mse_baseval.item()
            
    # Calculamos el error medio del examen
    avg_val_loss = val_loss_acumulada / len(val_loader)
    historial_val_loss.append(avg_val_loss)
    toc = time.time()

    # --- Supongamos que aquí termina tu validación de la época ---
# avg_val_loss es el promedio de pérdida de esta época en validación

    if avg_val_loss < best_val_loss:
        print(f"⭐ ¡Nuevo récord! La pérdida bajó de {best_val_loss:.6f} a {avg_val_loss:.6f}. Guardando...")
        
        # Actualizamos el récord
        best_val_loss = avg_val_loss
        
        # Creamos un diccionario con todo lo importante
        checkpoint = {
            'epoch': epoca,
            'model_state_dict': modelo.state_dict(),  # Los "pesos" de la IA
            'optimizer_state_dict': optimizador.state_dict(),
            'loss': best_val_loss,
            'stats': stats_norm # Incluimos las estadísticas de normalización que creamos antes
        }
        
        # Guardamos el archivo (esto sobrescribirá el anterior, quedándote siempre con el mejor)
        torch.save(checkpoint, ruta_mejor_modelo)

    # Imprimimos el progreso al final de cada época
    print(f"Época [{epoca+1:02d}/{epocas}] | Train Loss (MSE): {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    print(f"Tiempo de la época: {toc - tic:.2f} segundos")

print("-" * 60)
print("✅ ¡Entrenamiento completado!")
tac=time.time()

#5. GUARDAMOS LOS HIPERPARÁMETROS Y ESTADÍSTICAS EN UN ARCHIVO JSON

# Creamos un diccionario con los hiperparámetros y estadísticas que queremos guardar
metadata = {}
metadata['input_size'] = features_totales_in
metadata['hidden_size'] = lstm_hidden_size
metadata['num_lstmlayers'] = num_lstmlayers
metadata['output_size'] = features_totales_out
metadata['learning_rate'] = learning_rate
metadata['train_time_seconds'] = tac - tek
metadata['best_val_loss'] = best_val_loss
metadata['date_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
metadata['epocs'] = epocas
metadata['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
metadata['train_split_percentage'] = 0.8
metadata['num_train_samples'] = len(train_loader.dataset)


# Guardar el diccionario con los hiperparámetros y estadísticas en un archivo JSON
with open(ruta_carpeta_guardar+'/hyperparams.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)



# 6. VISUALIZACIÓN DE RESULTADOS
# ==========================================
# 1. DIBUJAMOS LA GRÁFICA
# ==========================================
plt.figure(figsize=(10, 6))

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
plt.show()


df = pd.DataFrame()
df['epoch']=range(1, len(historial_train_loss)+1)
df['train_loss']=historial_train_loss
df['val_loss']=historial_val_loss
df.to_csv(ruta_carpeta_guardar+'/historial_perdidas.csv', index=False, sep=',', float_format='%.6e')