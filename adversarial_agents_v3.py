import os
import torch
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.bfloat16


import gc
import pandas as pd
import json
import re
import sys
import numpy as np
from typing import TypedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import warnings
from transformers import logging as hf_logging

# ==========================================
# SILENCIAR TODOS LOS AVISOS (WARNINGS)
# ==========================================
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

# ==========================================
# FORZAR A PANDAS A MOSTRAR TABLAS ENTERAS
# ==========================================
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', None)

# ============================
# 0. Carga de configuración
# ============================

def fixText(string):
    return "".join(string.splitlines())

def readConfig(configFile):
    rOutput = ""
    dataFile = ""
    executionMode = ""
    model = ""
    with open(configFile) as config:
        for line in config:
            if line.find("rOutput") != -1:
                aux = line.split('=')
                rOutput = fixText(aux[1])
            if line.find("dataFile") != -1:
                aux = line.split('=')
                dataFile = fixText(aux[1])
            if line.find("executionMode") != -1:
                aux = line.split('=')
                executionMode = fixText(aux[1])
            if line.find("targetModel") != -1:
                aux = line.split('=')
                model = fixText(aux[1])
    if rOutput == "" or dataFile == "" or executionMode == "" or model == "":
        raise Exception("Falta un parámetro en el archivo de configuración.")
    return rOutput, dataFile, executionMode, model

# ============================
# 1. Configuración Principal
# ============================

rOutput, data_file, execution_mode, model = readConfig(sys.argv[1])

MODEL_NAME = model 
if execution_mode == "binary":
    BINARY_MODE = True
elif execution_mode == "multiclass":
    BINARY_MODE = False

ROUNDS = 1000

# --- Carga y Limpieza de Datos ---
df = pd.read_csv(data_file)
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
y_train_raw = df_train.iloc[:, -1].astype(str)

if BINARY_MODE:
    y_train_raw = y_train_raw.apply(lambda x: "benign" if "benign" in x.lower() or "normal" in x.lower() else "attack")
else:
    y_train_raw = y_train_raw.str.lower()

CLASS_NAMES = sorted(y_train_raw.unique().tolist())

if not BINARY_MODE and "unknown" not in CLASS_NAMES:
    CLASS_NAMES.append("unknown")

CLASS_LIST_STR = ", ".join([c for c in CLASS_NAMES if c != "unknown"])

X_train = df_train.iloc[:, :-1].select_dtypes(include='number').astype('float32')

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_raw)

print("Entrenando clasificador base (Random Forest) y analizando relevancia...")
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train_encoded)
COLUMNS_REQUERIDAS = X_train.columns.tolist()

importancias = clf_rf.feature_importances_
ranking = sorted(zip(importancias, COLUMNS_REQUERIDAS), reverse=True)
COLUMNAS_CRITICAS = [col for imp, col in ranking[:10]]

print(f"Columnas CRÍTICAS protegidas (no modificables): {COLUMNAS_CRITICAS}\n")

y_true_hist, y_pred_hist = [], []

def safe_extract_json(text, fallback_key="prediction"):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            cleaned = {k: float(v) for k, v in data.items() 
                       if k in COLUMNS_REQUERIDAS and str(v).replace('.','',1).isdigit()}
            if fallback_key in data:
                val = str(data[fallback_key]).lower()
                if "unknown" in val:
                    # --- NUEVA LÓGICA INTELIGENTE ---
                    cleaned[fallback_key] = "attack" if BINARY_MODE else "unknown"
                else:
                    cleaned[fallback_key] = next((c for c in CLASS_NAMES if c in val), "benign")
            return cleaned
        return {}
    except: return {}

# ============================
# 2. Inicialización de Agentes
# ============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
)

# --- OPTIMIZACIÓN DE HIPERPARÁMETROS PPO ---
config_ppo = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1.41e-5,          
    batch_size=4,                   # Aumentado para estabilidad de gradientes
    mini_batch_size=1,              # Mantenido en 1 para que no consuma VRAM extra
    gradient_accumulation_steps=4,  # Acumulamos 4 pasos antes de aplicar cambios
    optimize_cuda_cache=True,       
    early_stopping=False,
    target_kl=0.1,                  
)

def create_agent(name):
    # 1. Cargar el modelo base normal
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16, 
        device_map={"": 0}, 
        trust_remote_code=True
    )

    # 2. EL BUSCADOR ESPECÍFICO PARA GEMMA
    # Si es un modelo moderno, buscará en text_config
    if hasattr(base_model.config, "hidden_size"):
        h_size = base_model.config.hidden_size
    elif hasattr(base_model.config, "text_config") and hasattr(base_model.config.text_config, "hidden_size"):
        h_size = base_model.config.text_config.hidden_size
    elif hasattr(base_model.config, "n_embd"):
        h_size = base_model.config.n_embd
    elif hasattr(base_model.config, "d_model"):
        h_size = base_model.config.d_model
    else:
        # Fallback de seguridad por si acaso
        h_size = getattr(base_model.config, "word_embed_proj_dim", 2048)

    # 3. Preparar LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )
    lora_config.autocast_adapter_dtype = False 

    # 4. Envolver en PEFT
    peft_model = get_peft_model(base_model, lora_config)

    # 5. EL PARCHE PARA TRL
    # Restauramos la configuración original
    peft_model.config = base_model.config
    # Engañamos a TRL inyectando el "hidden_size" del texto en la raíz general
    peft_model.config.hidden_size = h_size

    # 6. Construir PPO
    ppo_model = AutoModelForCausalLMWithValueHead(peft_model)
    ppo_model.is_peft_model = True

    return PPOTrainer(config_ppo, ppo_model, None, tokenizer)

ppo_atk = create_agent("attacker")
ppo_def = create_agent("defender")

# ============================
# 3. Nodos del Grafo (LangGraph)
# ============================
class AgentState(TypedDict):
    current_log: dict
    label: str
    atk_data: dict
    def_data: dict
    rf_prediction: str
    reward_atk: float
    reward_def: float
    success_atk: bool
    partial_success_atk: bool
    success_def: bool
    llm_label_pred: str

def node_attacker(state):
    log_data = state['current_log']
    real_label = state['label'].lower()
    
    if "benign" in real_label:
        return {"atk_data": {"query": None, "resp": None}, "current_log": log_data}

    FLEXIBLES_CRITICAS = [c for c in COLUMNAS_CRITICAS if "port" not in c.lower() and "proto" not in c.lower()]
    INMUTABLES = [c for c in COLUMNAS_CRITICAS if c not in FLEXIBLES_CRITICAS]
    EPSILON_ESTRICTO = 0.05 
    EPSILON_GENERAL = 0.20  

    prompt = (f"Modify this {real_label} log. "
              f"Constraint: Max {EPSILON_ESTRICTO*100}% change on critical features: {FLEXIBLES_CRITICAS}. "
              f"Forbidden to touch: {INMUTABLES}. "
              f"Return ONLY JSON: {log_data}")
    
    query_t = tokenizer.encode(prompt, return_tensors="pt").to(next(ppo_atk.model.parameters()).device)
    full_output = ppo_atk.generate(query_t.squeeze(), max_new_tokens=128)
    resp_t = full_output[:, query_t.shape[-1]:] 
    
    cambios = safe_extract_json(tokenizer.decode(resp_t.squeeze(), skip_special_tokens=True))
    new_log = log_data.copy()
    
    for col, val_new in cambios.items():
        if col in INMUTABLES: continue 
        
        v_orig = float(log_data[col])
        margin = EPSILON_ESTRICTO if col in FLEXIBLES_CRITICAS else EPSILON_GENERAL
        
        val_final = max(v_orig * (1 - margin), min(val_new, v_orig * (1 + margin)))
        new_log[col] = val_final
            
    return {"current_log": new_log, "atk_data": {"query": query_t, "resp": resp_t}}

def node_defender(state):
    log_final = state['current_log']
    input_df = pd.DataFrame([log_final]).reindex(columns=COLUMNS_REQUERIDAS, fill_value=0)
    rf_label = le.inverse_transform([clf_rf.predict(input_df)[0]])[0]
    
    mode_desc = "is this benign or attack?" if BINARY_MODE else f"classify: {CLASS_LIST_STR}"
    prompt = (f"You are a strict cybersecurity expert. Analyze this network log: {log_final}. "
              f"{mode_desc}. You MUST respond ONLY with a valid JSON. No explanations, no extra text. "
              f"Format strictly like this: {{\"prediction\": \"your_class\"}}")
    
    query_t = tokenizer.encode(prompt, return_tensors="pt").to(next(ppo_def.model.parameters()).device)
    full_output = ppo_def.generate(query_t.squeeze(), max_new_tokens=100)
    resp_t = full_output[:, query_t.shape[-1]:]
    msg = tokenizer.decode(resp_t.squeeze(), skip_special_tokens=True)
    
    print(f"\n[RAW DEFENDER MSG] -> {msg}\n")

    return {
        "def_data": {"query": query_t, "resp": resp_t, "msg": msg}, 
        "rf_prediction": rf_label,
        "llm_label_pred": safe_extract_json(msg).get("prediction", "benign")
    }

def node_evaluator(state):
    real, llm_pred, rf_pred = state['label'].lower(), state['llm_label_pred'], state['rf_prediction']
    
    def_success = (llm_pred == real)
    atk_success = ("benign" not in real) and (llm_pred == "benign" and rf_pred == "benign")
    partial_atk_success = ("benign" not in real) and (llm_pred == "benign" or rf_pred == "benign")
    
    attack_reward = -0.4
    if partial_atk_success: attack_reward = 0.25
    if atk_success: attack_reward = 0.5
    
    return {
        "reward_atk": attack_reward, 
        "reward_def": 0.5 if def_success else -0.25, 
        "success_atk": atk_success, "partial_success_atk": partial_atk_success, "success_def": def_success
    }

workflow = StateGraph(AgentState)
workflow.add_node("attacker", node_attacker)
workflow.add_node("defender", node_defender)
workflow.add_node("evaluator", node_evaluator)
workflow.set_entry_point("attacker")
workflow.add_edge("attacker", "defender")
workflow.add_edge("defender", "evaluator")
workflow.add_edge("evaluator", END)
app = workflow.compile()

# ============================
# 4. Entrenamiento RL y Reporte
# ============================
print(f"SISTEMA INICIADO | Modelo: {MODEL_NAME} | Modo: {'Binario' if BINARY_MODE else 'Multiclase'}\n")
print(f"Iniciando RL con {ROUNDS} muestras aleatorias del conjunto de TEST (no vistas previamente)...")

# --- NUEVO: Listas para acumular lotes (Batches) ---
BATCH_SIZE = 4
batch_queries_atk, batch_resps_atk, batch_rewards_atk = [], [], []
batch_queries_def, batch_resps_def, batch_rewards_def = [], [], []

for i in range(ROUNDS):
    row = df_test.sample(n=1).iloc[0]
    
    label_val = str(row.iloc[-1]).lower()
    if BINARY_MODE and "benign" not in label_val: 
        label_val = "attack"
    
    state = app.invoke({"current_log": row[:-1].to_dict(), "label": label_val})
    
    y_true_hist.append(state['label'])
    y_pred_hist.append(state['llm_label_pred'])
    
    # Acumulamos los datos generados en este turno
    if state["atk_data"]["query"] is not None:
        batch_queries_atk.append(state["atk_data"]["query"].squeeze(0))
        batch_resps_atk.append(state["atk_data"]["resp"].squeeze(0))
        batch_rewards_atk.append(torch.tensor(state["reward_atk"], dtype=torch.float))
        
    batch_queries_def.append(state["def_data"]["query"].squeeze(0))
    batch_resps_def.append(state["def_data"]["resp"].squeeze(0))
    batch_rewards_def.append(torch.tensor(state["reward_def"], dtype=torch.float))

    # --- ENTRENAMIENTO ATACANTE (Solo si ha acumulado 4 ataques) ---
    if len(batch_queries_atk) == BATCH_SIZE:
        device_atk = ppo_atk.accelerator.device
        b_q_atk = [q.to(device_atk) for q in batch_queries_atk]
        b_r_atk = [r.to(device_atk) for r in batch_resps_atk]
        b_rw_atk = [rw.to(device_atk) for rw in batch_rewards_atk]
        
        ppo_atk.step(b_q_atk, b_r_atk, b_rw_atk)
        
        # Vaciamos sus listas
        batch_queries_atk.clear()
        batch_resps_atk.clear()
        batch_rewards_atk.clear()
        
        # Limpieza de memoria exclusiva para el atacante
        gc.collect()
        torch.cuda.empty_cache()

    # --- ENTRENAMIENTO DEFENSOR (Cada 4 rondas de cualquier tráfico) ---
    if len(batch_queries_def) == BATCH_SIZE:
        device_def = ppo_def.accelerator.device
        b_q_def = [q.to(device_def) for q in batch_queries_def]
        b_r_def = [r.to(device_def) for r in batch_resps_def]
        b_rw_def = [rw.to(device_def) for rw in batch_rewards_def]
        
        ppo_def.step(b_q_def, b_r_def, b_rw_def)
        
        # Vaciamos sus listas
        batch_queries_def.clear()
        batch_resps_def.clear()
        batch_rewards_def.clear()
        
        # Limpieza de memoria exclusiva para el defensor
        gc.collect()
        torch.cuda.empty_cache()

# --- Reporte Final ---
print(f"\n{'='*60}\n MATRIZ DE CONFUSIÓN (DEFENSOR LLM)\n{'='*60}")
cm = confusion_matrix(y_true_hist, y_pred_hist, labels=CLASS_NAMES)
df_cm = pd.DataFrame(cm, index=[f"Real_{c}" for c in CLASS_NAMES], columns=[f"Pred_{c}" for c in CLASS_NAMES])
print(df_cm)

print("\nInforme de Clasificación:")
reporte_clas = classification_report(y_true_hist, y_pred_hist, labels=CLASS_NAMES, zero_division=0)
print(reporte_clas)

# --- NUEVO: Guardar los resultados en el archivo rOutput ---
try:
    with open(rOutput, "w", encoding="utf-8") as f_out:
        f_out.write(f"{'='*60}\n MATRIZ DE CONFUSIÓN (DEFENSOR LLM)\n{'='*60}\n")
        f_out.write(df_cm.to_string())
        f_out.write("\n\nInforme de Clasificación:\n")
        f_out.write(reporte_clas)
    print(f"\n[INFO] Resultados guardados con éxito en el archivo: {rOutput}")
except Exception as e:
    print(f"\n[ERROR] No se pudo guardar el reporte en {rOutput}. Detalle: {e}")
