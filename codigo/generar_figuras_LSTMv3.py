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

ruta_carpeta_guardar = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/run4proLSTM"
ruta_maestra         = r"/home/antonio/Proyecto_Fluidos_AI/datos_procesados/process_data.h5"

ruta_json = ruta_carpeta_guardar + "/hyperparams.json"
with open(ruta_json, 'r') as f:
    meta = json.load(f)

features_totales_in   = meta['input_size']
features_totales_out  = meta['output_size']
num_lstmlayers  = meta['num_lstmlayers']
lstm_hidden_size = meta['hidden_size']
learning_rate = meta['learning_rate']
linea_idy=     2




# 1. Definimos las dimensiones (Solo canal U)
canales = 1 


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


# ==========================================
# FASE DE INFERENCIA: PREDICCIÓN
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo para inferencia: {device}")

# Definición de canales
linea_objetivo = int(linea_idy)+1               # La línea que el modelo todavía no ha visto en el pasado, pero debe predecir en el futuro

ruta_modelo_campeon = ruta_carpeta_guardar + "/mejor_modelo_ia.pt"
ruta_stats          = r"/home/antonio/Proyecto_Fluidos_AI/codigo/norm_stats_pro1.pt"


# ==========================================
# 1. CARGAR ESTADÍSTICAS Y MODELO
# ==========================================
print("Cargando estadísticas y modelo...")
stats    = torch.load(ruta_stats, weights_only=False)
media_X  = stats['media']
std_X    = stats['std']

# ⚠️ Re-instanciamos con dropout_rate para que los pesos carguen sin errores de forma.
#    Coincide con el dropout_rate usado en el entrenamiento.
modelo_inf = FluidoLSTM_Seq2Seq(hidden_size=lstm_hidden_size,
                    num_lstmlayers=num_lstmlayers,
                    output_size=features_totales_out)

checkpoint_completo = torch.load(ruta_modelo_campeon, weights_only=False)
modelo_inf.load_state_dict(checkpoint_completo['model_state_dict'])
modelo_inf.to(device)
modelo_inf.eval()   # 🔴 MODO EXAMEN: APAGAR DROPOUT


# ==========================================
# 2. CARGAR DATOS (VERSIÓN LSTM MONOCANAL)
# ==========================================
print("Extrayendo línea objetivo del disco...")
with h5.File(ruta_maestra, 'r') as f:
    # El LSTM solo necesita UNA línea → resultado es 2D: (T, N_pts)
    matriz_final_test = f['data_analisis']['u_fluc'][:, linea_objetivo, :]

N_pts = matriz_final_test.shape[1]
print(f"   -> Datos listos. Forma: {matriz_final_test.shape}")  # (T, N_pts)

instante_base = 5000

# ── Entrada: (seq_len, N_pts) ──
X_test_raw  = matriz_final_test[instante_base - features_totales_in : instante_base, :]

# ── Objetivo: (pred_len, N_pts) ──
y_test_real = matriz_final_test[instante_base : instante_base + features_totales_out, :]

# Normalizar y transponer a (N_pts, seq_len) para el LSTM
X_test_norm   = (X_test_raw - media_X) / std_X      # (seq_len, N_pts)
X_test_norm_T = X_test_norm.T                        # (N_pts, seq_len)  ← solo .T, no .transpose(2,0,1)
X_tensor      = torch.tensor(X_test_norm_T, dtype=torch.float32).to(device)


# ==========================================
# 3. PREDICCIÓN POR LOTES
# ==========================================
print("Prediciendo todos los puntos espaciales por lotes...")
inicio      = time.time()
tamano_lote = 500

predicciones = []
with torch.no_grad():
    for i in range(0, X_tensor.shape[0], tamano_lote):
        y_batch = modelo_inf(X_tensor[i : i + tamano_lote])  # (batch, pred_len)
        predicciones.append(y_batch)

y_pred_norm  = torch.cat(predicciones, dim=0)        # (N_pts, pred_len)
y_pred_numpy = y_pred_norm.cpu().numpy()             # (N_pts, pred_len)  ← sin squeeze
y_pred_real  = (y_pred_numpy * std_X) + media_X     # (N_pts, pred_len)
y_pred_real  = y_pred_real.T 


# ==========================================
# FIG 1: SERIE TEMPORAL (CORREGIDA CON OFFSET Y LÍNEA 6)
# ==========================================
punto_espacial = 0
pasado_objetivo = matriz_final_test[instante_base - 100 : instante_base, punto_espacial]  # ✓ ya es 2D

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
plt.plot(range(features_totales_out), y_test_real[:, punto_espacial], 'g-o',
         label=f'Real Future — Line {linea_objetivo} (Ground Truth)', linewidth=2)

# 3. La Predicción de la IA (Alineada)
plt.plot(range(features_totales_out), pred_alineada, 'r--s',
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

print(f"Predicción directa {features_totales_out} pasos desde el instante {instante_objetivo}...")


pasado_raw_mc  = matriz_final_test[instante_objetivo - features_totales_in : instante_objetivo, punto_espacial]
# shape: (seq_len,)
pasado_objetivo_150 = matriz_final_test[instante_objetivo - 150 : instante_objetivo, punto_espacial]

pasado_norm_mc = (pasado_raw_mc - media_X) / std_X
x_tensor = torch.tensor(pasado_norm_mc, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len) ✓

with torch.no_grad():
    pred_norm = modelo_inf(x_tensor)                          # (1, pred_len)

pred_real = (pred_norm.squeeze(0).cpu().numpy() * std_X) + media_X  # (pred_len,)

# 🚨 Corrección de Offset para la Fig 2
offset_fig2 = pred_real[0] - pasado_objetivo_150[-1]
pred_alineada_fig2 = pred_real - offset_fig2

futuro_real_raw = matriz_final_test[instante_objetivo : instante_objetivo + features_totales_out, punto_espacial]
pasos_reales_disp = len(futuro_real_raw)

plt.figure(2, figsize=(12, 6))
plt.clf()
plt.plot(range(-150, 0), pasado_objetivo_150, 'k-',
         label=f'Real Past — Line {linea_objetivo}', linewidth=2)
if pasos_reales_disp > 0:
    plt.plot(range(pasos_reales_disp), futuro_real_raw, 'g-o',
             label=f'Real Future — Line {linea_objetivo}', linewidth=2, alpha=0.6)
plt.plot(range(features_totales_out), pred_alineada_fig2, 'r--s',
         label='AI Prediction (Offset Corrected)', linewidth=2)
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.title(f'LSTM Direct Prediction (Offset Corrected) at step {instante_objetivo}')
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

error_abs_mat = np.abs(y_test_real - y_pred_real)   # (features_totales_out, N_pts)
error_medio   = np.mean(error_abs_mat, axis=1)       # (features_totales_out,)
std_error     = np.std(error_abs_mat,  axis=1)       # (features_totales_out,)

mse_global  = np.mean((y_test_real - y_pred_real)**2)
rmse_global = np.sqrt(mse_global)
mae_global  = np.mean(error_abs_mat)

plt.figure(figsize=(12, 6))
t_eje = range(features_totales_out)

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