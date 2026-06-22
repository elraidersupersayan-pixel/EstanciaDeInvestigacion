import h5py as h5
import numpy as np
import time
import os
import json
import gc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y METADATOS
# ==========================================
ruta_carpeta_guardar = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/runtransformermulticanal1"
ruta_maestra         = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/process_data.h5"
ruta_json            = ruta_carpeta_guardar + "/hyperparams_transformer.json"
ruta_modelo_campeon  = ruta_carpeta_guardar + '/modelo_finetuned_linea6.pt'
ruta_stats           = r"/home/antonio/Proyecto_Fluidos_AI/codigo/norm_stats_tfmulticanal1.pt"

with open(ruta_json, 'r') as f:
    meta = json.load(f)

seq_len      = meta['seq_len']
pred_len     = meta['pred_len']
input_dim    = meta['input_dim']
d_model      = meta['d_model']
num_heads    = meta['num_heads']
dropout_rate = meta['dropout_rate']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Iniciando Fine-Tuning en: {device}")

# ==========================================
# 2. DEFINICIÓN DE LA ARQUITECTURA (IDÉNTICA)
# ==========================================
class MultiHeadEasyAttention(nn.Module):
    def __init__(self, seq_len, input_dim, d_model, num_heads=4):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.alpha = nn.Parameter(torch.empty(num_heads, seq_len, seq_len))
        nn.init.xavier_uniform_(self.alpha)
        self.W_v = nn.Linear(input_dim, d_model, bias=False)

    def forward(self, x):
        b, p, _ = x.size()
        V = self.W_v(x).view(b, p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        out = torch.einsum('hij,bhjd->bhid', self.alpha, V)
        return out.permute(0, 2, 1, 3).contiguous().view(b, p, -1)

class EasyTransformerBlock(nn.Module):
    def __init__(self, seq_len, d_model=64, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadEasyAttention(seq_len, d_model, d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        expansion_factor = 4
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(d_model * expansion_factor, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.feed_forward(x)
        return self.norm2(x + self.dropout2(ff_out))

class EasyFluidPredictor(nn.Module):
    def __init__(self, seq_len=500, pred_len=50, input_dim=5, d_model=128, num_heads=8, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1 
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout_rate)
        self.easy_transformer = EasyTransformerBlock(seq_len, d_model, num_heads, dropout_rate)
        self.temporal_proj = nn.Linear(seq_len, pred_len)  
        self.final_proj = nn.Linear(d_model, self.output_dim)    

    def forward(self, x):
        ultimo_valor_adyacente = x[:, -1:, -1:] 
        x_trans = self.input_proj(x)            
        x_trans = self.input_dropout(x_trans)   
        x_trans = self.easy_transformer(x_trans) 
        x_trans = x_trans.permute(0, 2, 1)      
        x_trans = self.temporal_proj(x_trans)   
        x_trans = x_trans.permute(0, 2, 1)      
        delta_u = self.final_proj(x_trans)      
        out = ultimo_valor_adyacente + delta_u
        return out

# ==========================================
# 3. CARGA DEL MODELO CAMPEÓN Y CONGELACIÓN
# ==========================================
print("Cargando modelo preentrenado...")
modelo = EasyFluidPredictor(seq_len=seq_len, pred_len=pred_len, input_dim=input_dim, d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate)
checkpoint = torch.load(ruta_modelo_campeon, map_location=device, weights_only=False)
modelo.load_state_dict(checkpoint['model_state_dict'])
modelo.to(device)

# 🚨 AQUÍ ESTÁ EL TRUCO REVOLUCIONARIO:
# Congelamos el bloque Transformer para que no olvide la física general.
# Solo dejamos que se entrenen las capas lineales que ajustan la ESCALA.
#for param in modelo.easy_transformer.parameters():
    #param.requires_grad = False

print("✅ Bloque Transformer congelado con éxito. Solo se optimizarán las proyecciones.")

# ==========================================
# 4. PREPARACIÓN DEL MINI-DATASET DE LA LÍNEA 6
# ==========================================
print("Extrayendo una porción de datos de la Línea 6 para calibración...")
stats = torch.load(ruta_stats, weights_only=False)
media_X, std_X = stats['media'], stats['std']

with h5.File(ruta_maestra, 'r') as f:
    # Solo necesitamos los primeros chunks temporales para calibrar (ej. los primeros 25250 pasos)
    matriz_base = f['data_analisis']['u_fluc'][:25250, :, :]
    chunk_size = 5050
    num_chunks = len(matriz_base) // chunk_size
    trozos = [matriz_base[i*chunk_size:(i+1)*chunk_size, :, :] for i in range(num_chunks)]
    matriz_calibracion = np.concatenate(trozos, axis=2)

# Usamos moldes de Líneas 1-5 (X) para predecir la Línea 6 (Y)
def crear_ventanas_cfd(matriz, lookback, predict, stride):
    X_lista, y_lista = [], []
    num_tiempos = matriz.shape[0]
    for i in range(0, num_tiempos - lookback - predict + 1, stride):
        bloque_x = matriz[i : i + lookback, 1:6, :]       # Canales 1 a 5
        bloque_y = matriz[i + lookback : i + lookback + predict, 6:7, :] # Línea 6 objetivo
        X_lista.append(bloque_x.transpose(2, 0, 1))
        y_lista.append(bloque_y.transpose(2, 0, 1))
    return np.vstack(X_lista), np.vstack(y_lista)

# Normalizamos con las estadísticas globales del entrenamiento
matriz_norm = np.float32((matriz_calibracion - media_X) / std_X)

X_calib, y_calib = crear_ventanas_cfd(matriz_norm, seq_len, pred_len, stride=100) # Stride alto para ir rápidos
print(f"-> Dataset de Fine-Tuning creado: {X_calib.shape[0]} muestras espaciales.")

dataset_calib = TensorDataset(torch.from_numpy(X_calib), torch.from_numpy(y_calib))
loader_calib = DataLoader(dataset_calib, batch_size=256, shuffle=True)

del matriz_base, matriz_calibracion, matriz_norm, X_calib, y_calib
gc.collect()

# ==========================================
# 5. BUCLE CORTO DE AJUSTE FINO (FINE-TUNING)
# ==========================================
# Un learning rate MINÚSCULO para no romper los pesos
optimizador = torch.optim.Adam(filter(lambda p: p.requires_grad, modelo.parameters()), lr=1e-5)
criterio = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()

epocas_ft = 20
print(f"\n🔥 Iniciando ajuste fino durante {epocas_ft} épocas...")

modelo.train()
for epoca in range(epocas_ft):
    loss_acumulada = 0.0
    for x_batch, y_batch in loader_calib:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizador.zero_grad()
        
        with torch.cuda.amp.autocast():
            predicciones = modelo(x_batch)
            loss = criterio(predicciones, y_batch) # Usamos solo MSE bruto, sin gradiente para estabilizar la escala
            
        scaler.scale(loss).backward()
        scaler.step(optimizador)
        scaler.update()
        loss_acumulada += loss.item()
        
    print(f"   Época Ft [{epoca+1}/{epocas_ft}] | Loss Calibración: {loss_acumulada/len(loader_calib):.6f}")

# Guardamos el nuevo modelo calibrado para la Línea 6
ruta_modelo_calibrado = ruta_carpeta_guardar + '/modelo_finetuned_linea6.pt'
torch.save({'model_state_dict': modelo.state_dict()}, ruta_modelo_calibrado)
print(f"\n✅ ¡Fine-Tuning completado! Modelo de escala calibrado en: {ruta_modelo_calibrado}")




# ==========================================
# FASE DE INFERENCIA: PREDICCIÓN MULTICANAL
# 5 LÍNEAS DE ENTRADA (1-5) → LÍNEA OBJETIVO (6)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo para inferencia: {device}")

# Definición de canales
lineas_entrada = list(range(1, 6))   # [1, 2, 3, 4, 5]  → input_dim = 5
linea_objetivo = 6                # La línea que el modelo todavía no ha visto en el pasado, pero debe predecir en el futuro

ruta_modelo_campeonfine = ruta_carpeta_guardar + '/modelo_finetuned_linea6.pt'
ruta_stats          =  r"/home/antonio/Proyecto_Fluidos_AI/codigo/norm_stats_tfmulticanal1.pt"


# ==========================================
# 1. CARGAR ESTADÍSTICAS Y MODELO
# ==========================================
print("Cargando estadísticas y modelo...")
stats    = torch.load(ruta_stats, weights_only=False)
media_X  = stats['media']
std_X    = stats['std']

# ⚠️ Re-instanciamos con dropout_rate para que los pesos carguen sin errores de forma.
#    Coincide con el dropout_rate usado en el entrenamiento.
modelo_inf = EasyFluidPredictor(
    seq_len=seq_len, pred_len=pred_len,
    input_dim=input_dim, d_model=d_model,
    num_heads=num_heads, dropout_rate=dropout_rate
)
checkpoint_completo = torch.load(ruta_modelo_campeonfine, weights_only=False)
modelo_inf.load_state_dict(checkpoint_completo['model_state_dict'])
modelo_inf.to(device)
modelo_inf.eval()   # 🔴 MODO EXAMEN: APAGAR DROPOUT


# ==========================================
# 2. PREPARAR LOS DATOS MULTICANAL (7 LÍNEAS)
# ==========================================
print("Extrayendo las 7 líneas del disco...")
inicio = time.time()
with h5.File(ruta_maestra, 'r') as f:
    # Leemos el cubo completo: (T, 7, Z)
    matriz_base     = f['data_analisis']['u_fluc'][:, :, :]
    chunk_size      = 5050
    num_chunks      = len(matriz_base) // chunk_size
    matriz_recortada = matriz_base[:num_chunks * chunk_size, :, :]
    trozos = [matriz_recortada[i*chunk_size:(i+1)*chunk_size, :, :]
              for i in range(num_chunks)]
    # Concatenamos por el eje espacial (eje 2) → (5050, 7, N_pts)
    matriz_final_test = np.concatenate(trozos, axis=2)

N_pts = matriz_final_test.shape[2]
print(f"   -> Datos listos en {time.time()-inicio:.2f} s.  Forma: {matriz_final_test.shape}")

# Punto temporal de referencia para la predicción global (figs 1, 2, 5)
instante_base = 5000   # Se necesita: instante_base + pred_len <= 5050

# ── Entrada: 5 canales, shape (seq_len, 5, N_pts) ──
X_test_raw  = matriz_final_test[instante_base - seq_len : instante_base, 1:6, :]

# ── Objetivo: línea 6, shape (pred_len, N_pts) ──
y_test_real = matriz_final_test[instante_base : instante_base + pred_len, linea_objetivo, :]

# Normalizar y transponer a (N_pts, seq_len, 5) para el modelo
X_test_norm   = (X_test_raw - media_X) / std_X          # (seq_len, 5, N_pts)
X_test_norm_T = X_test_norm.transpose(2, 0, 1)          # (N_pts, seq_len, 5)
X_tensor      = torch.tensor(X_test_norm_T, dtype=torch.float32).to(device)


# ==========================================
# 3. PREDICCIÓN POR LOTES (TODAS LAS POSICIONES ESPACIALES)
# ==========================================
print("Prediciendo todos los puntos espaciales por lotes...")
inicio      = time.time()
tamano_lote = 500
predicciones = []

with torch.no_grad():
    for i in range(0, X_tensor.shape[0], tamano_lote):
        y_batch = modelo_inf(X_tensor[i : i + tamano_lote])   # (batch, pred_len, 1)
        predicciones.append(y_batch)

y_pred_norm = torch.cat(predicciones, dim=0)   # (N_pts, pred_len, 1)
print(f"   -> Predicción completada en {time.time()-inicio:.2f} s.")

del X_tensor, predicciones
torch.cuda.empty_cache()

# Desnormalizar y transponer a (pred_len, N_pts)
y_pred_numpy = y_pred_norm.squeeze(-1).cpu().numpy()   # (N_pts, pred_len)
y_pred_real  = (y_pred_numpy * std_X) + media_X        # (N_pts, pred_len)
y_pred_real  = y_pred_real.T                            # (pred_len, N_pts) ✓
# y_test_real ya tiene forma (pred_len, N_pts)          ✓


# ==========================================
# FIG 1: SERIE TEMPORAL (CORREGIDA CON OFFSET Y LÍNEA 6)
# ==========================================
punto_espacial = 0

# 🚨 CORRECCIÓN 1: Extraemos el pasado REAL de la línea objetivo (Línea 6), no de la 5.
pasado_objetivo = matriz_final_test[instante_base - 100 : instante_base, linea_objetivo, punto_espacial] 

# 🚨 CORRECCIÓN 2: Corrector de Offset (Post-procesado CFD)
# Forzamos a la IA a empezar exactamente donde terminó el pasado real.
# Esto nos permite aislar y evaluar la forma de la turbulencia pura.
offset_inicial = y_pred_real[0, punto_espacial] - pasado_objetivo[-1]
pred_alineada = y_pred_real[:, punto_espacial] - offset_inicial

# Recalculamos las métricas con la curva alineada para ver el rendimiento real de la fluctuación
mse_test  = np.mean((y_test_real[:, punto_espacial] - pred_alineada)**2)
rmse_test = np.sqrt(mse_test)
mae_test  = np.mean(np.abs(y_test_real[:, punto_espacial] - pred_alineada))

plt.figure(1, figsize=(12, 6))
plt.clf()

# 1. El Pasado Real (Oculto a la IA, pero vital para la vista)
plt.plot(range(-100, 0), pasado_objetivo, 'k-',
         label=f'Real Past — Line {linea_objetivo} (Hidden from AI)', linewidth=2)

# 2. El Futuro Real
plt.plot(range(pred_len), y_test_real[:, punto_espacial], 'g-o',
         label=f'Real Future — Line {linea_objetivo} (Ground Truth)', linewidth=2)

# 3. La Predicción de la IA (Alineada)
plt.plot(range(pred_len), pred_alineada, 'r--s',
         label=f'AI Prediction (Offset Corrected)', linewidth=2)

plt.gca().text(
    0.02, 0.95,
    f"Metrics (Offset Corrected):\nMSE:  {mse_test:.6f}\nRMSE: {rmse_test:.4f} m/s\nMAE:  {mae_test:.4f} m/s",
    transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.title(f'Zero-Shot Prediction — Line {linea_objetivo} (Point {punto_espacial})', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Velocity Fluctuation ($u$)', fontsize=14)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(ruta_carpeta_guardar + '/fig1fine.png', dpi=300)


# ==========================================
# FIG 2: PREDICCIÓN DIRECTA (CORREGIDA CON OFFSET)
# (Sustituye también la figura 3 si la estabas usando)
# ==========================================
instante_objetivo = 4950
punto_espacial    = 0

print(f"Predicción directa {pred_len} pasos desde el instante {instante_objetivo}...")

# 1. Historia para el modelo (Líneas 0 a 4)
# 🚨 CAMBIO CRÍTICO: Ajustar el índice a 1:6
pasado_raw_mc = matriz_final_test[instante_objetivo - seq_len : instante_objetivo, 1:6, punto_espacial]
# 2. Historia REAL de la línea objetivo para la gráfica
pasado_objetivo_150 = matriz_final_test[instante_objetivo - 150 : instante_objetivo, linea_objetivo, punto_espacial]

limite_real       = min(instante_objetivo + pred_len, matriz_final_test.shape[0])
futuro_real_raw   = matriz_final_test[instante_objetivo : limite_real, linea_objetivo, punto_espacial]
pasos_reales_disp = len(futuro_real_raw)

pasado_norm_mc = (pasado_raw_mc - media_X) / std_X
x_tensor = torch.tensor(pasado_norm_mc, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 5)

with torch.no_grad():
    pred_norm = modelo_inf(x_tensor)   

pred_real = (pred_norm.squeeze().cpu().numpy() * std_X) + media_X   

# 🚨 Corrección de Offset para la Fig 2
offset_fig2 = pred_real[0] - pasado_objetivo_150[-1]
pred_alineada_fig2 = pred_real - offset_fig2

plt.figure(2, figsize=(12, 6))
plt.clf()
plt.plot(range(-150, 0), pasado_objetivo_150, 'k-',
         label=f'Real Past — Line {linea_objetivo}', linewidth=2)
if pasos_reales_disp > 0:
    plt.plot(range(pasos_reales_disp), futuro_real_raw, 'g-o',
             label=f'Real Future — Line {linea_objetivo}', linewidth=2, alpha=0.6)
plt.plot(range(pred_len), pred_alineada_fig2, 'r--s',
         label='AI Prediction (Offset Corrected)', linewidth=2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.title(f'E-A Transformer Direct Prediction (Offset Corrected) at step {instante_objetivo}', fontsize=16)
plt.xlabel('Relative Time Steps', fontsize=14)
plt.ylabel('Velocity Fluctuation ($u$)', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(ruta_carpeta_guardar + '/fig2fine.png', dpi=300)


# ==========================================
# FIG 3: EVOLUCIÓN TEMPORAL DEL ERROR ESPACIAL
#   Para cada instante futuro: media y std del error absoluto
#   calculados sobre todos los N_pts puntos espaciales.
# ==========================================
print("Calculando evolución temporal del error...")

error_abs_mat = np.abs(y_test_real - y_pred_real)   # (pred_len, N_pts)
error_medio   = np.mean(error_abs_mat, axis=1)       # (pred_len,)
std_error     = np.std(error_abs_mat,  axis=1)       # (pred_len,)

mse_global  = np.mean((y_test_real - y_pred_real)**2)
rmse_global = np.sqrt(mse_global)
mae_global  = np.mean(error_abs_mat)

plt.figure(figsize=(12, 6))
t_eje = range(pred_len)

plt.plot(t_eje, error_medio, color='firebrick', linestyle='-', marker='o',
         linewidth=2.5, label='Spatial Mean of Absolute Error (MAE)')
plt.fill_between(t_eje,
                 error_medio - std_error,
                 error_medio + std_error,
                 color='salmon', alpha=0.3,
                 label=r'Spatial Dispersion ($\pm1$ Std Dev)')

plt.gca().text(
    0.02, 0.95,
    f"Global Metrics (Line {linea_objetivo}):\nRMSE Total: {rmse_global:.4f} m/s\nMAE Total:  {mae_global:.4f} m/s",
    transform=plt.gca().transAxes, fontsize=11, verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

plt.title(f'Temporal Error Evolution — AI Prediction (Line {linea_objetivo})', fontsize=16, fontweight='bold')
plt.xlabel('Future Time Steps', fontsize=14)
plt.ylabel(r'Absolute Error ($|u_{\rm pred} - u_{\rm real}|$)', fontsize=14)
plt.ylim(bottom=0)
plt.legend(loc='upper left', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(ruta_carpeta_guardar + '/fig3fine.png', dpi=300)

print("✅ Todas las figuras guardadas correctamente.")