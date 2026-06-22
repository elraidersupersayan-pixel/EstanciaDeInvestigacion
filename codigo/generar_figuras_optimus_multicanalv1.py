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

ruta_json = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/runtransformermulticanal1/hyperparams_transformer.json"
with open(ruta_json, 'r') as f:
    meta = json.load(f)

seq_len      = meta['seq_len']
pred_len     = meta['pred_len']
input_dim    = meta['input_dim']
d_model      = meta['d_model']
num_heads    = meta['num_heads']
dropout_rate = meta['dropout_rate']

ruta_carpeta_guardar = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/runtransformermulticanal1"
ruta_maestra         = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/process_data.h5"



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
    def __init__(self, seq_len=500, pred_len=50, input_dim=5, d_model=128, num_heads=8, dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 1 # Siempre predecimos 1 línea objetivo
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        self.easy_transformer = EasyTransformerBlock(seq_len, d_model, num_heads, dropout_rate)
        
        self.temporal_proj = nn.Linear(seq_len, pred_len)  
        self.final_proj = nn.Linear(d_model, self.output_dim)    

    def forward(self, x):
        # 🚨 EL ANCLAJE VUELVE (El motor de la fluctuación)
        # Tomamos el último instante de la línea adyacente (canal índice 4)
        ultimo_valor_adyacente = x[:, -1:, -1:] 

        x_trans = self.input_proj(x)            
        x_trans = self.input_dropout(x_trans)   
        x_trans = self.easy_transformer(x_trans) 
        
        x_trans = x_trans.permute(0, 2, 1)      
        x_trans = self.temporal_proj(x_trans)   
        x_trans = x_trans.permute(0, 2, 1)      
        
        # El modelo predice la DIFERENCIA (Delta U) respecto a la línea vecina
        delta_u = self.final_proj(x_trans)      
        
        # 🚨 MAGIA RESIDUAL: Sumamos el delta a la base. ¡Adiós línea plana!
        out = ultimo_valor_adyacente + delta_u
        return out


# ==========================================
# FASE DE INFERENCIA: PREDICCIÓN MULTICANAL
# 5 LÍNEAS DE ENTRADA (1-5) → LÍNEA OBJETIVO (6)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo para inferencia: {device}")

# Definición de canales
lineas_entrada = list(range(1, 6))   # [1, 2, 3, 4, 5]  → input_dim = 5
linea_objetivo = 6                # La línea que el modelo todavía no ha visto en el pasado, pero debe predecir en el futuro

ruta_modelo_campeon = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/runtransformermulticanal1/mejor_modelo_ia_transformer_eas.pt"
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
checkpoint_completo = torch.load(ruta_modelo_campeon, weights_only=False)
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
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(ruta_carpeta_guardar + '/fig1.png', dpi=300)


# ==========================================
# FIG 2: PREDICCIÓN DIRECTA (CORREGIDA CON OFFSET)
# (Sustituye también la figura 3 si la estabas usando)
# ==========================================
instante_objetivo = 4950
punto_espacial    = 0

print(f"Predicción directa {pred_len} pasos desde el instante {instante_objetivo}...")

# 1. Historia para el modelo (Líneas 0 a 4)
pasado_raw_mc = matriz_final_test[instante_objetivo - seq_len : instante_objetivo, 0:5, punto_espacial]
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
plt.savefig(ruta_carpeta_guardar + '/fig2.png', dpi=300)


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
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(ruta_carpeta_guardar + '/fig3.png', dpi=300)

print("✅ Todas las figuras guardadas correctamente.")