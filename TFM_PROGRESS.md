# TFM - Chatbot IA para Plamily

> Trabajo Final de Master - Big Data e IA
> Autor: Pol Ballarin Costa

---

## Objetivo

Crear un chatbot inteligente integrado en Plamily que permita a los usuarios ejecutar acciones mediante lenguaje natural (texto y voz). Enfoque hibrido: modelo propio para intent classification + NER, con Claude API como complemento para gestion de dialogo.

---

## Arquitectura

```
Flutter (Chat UI) → Cloud Function / Cloud Run (modelo + logica) → Firestore (ejecuta accion)
```

### Pipeline de procesamiento
1. Usuario escribe/dicta una frase
2. **Modelo propio** clasifica el intent y extrae entidades
3. Si faltan campos obligatorios → **Claude API** genera pregunta natural
4. Cuando todos los campos estan completos → ejecutar accion en Firestore
5. Confirmar al usuario

### Fases del proyecto
1. Generacion de dataset (castellano)
2. Entrenamiento y comparacion de modelos
3. Endpoint backend (Cloud Function / Cloud Run)
4. Chat UI en Flutter
5. STT / TTS (capa de voz)
6. Futuro: catalan, otros idiomas, Alexa, Siri

---

## Intents y Entidades

### Intents definidos (12)

| Intent | Campos obligatorios (se preguntan si faltan) | Campos opcionales | Objetivo | Generados |
|--------|----------------------------------------------|-------------------|----------|-----------|
| `add_medication` | medication_name, person, dose, frequency, time | — | 20,000 | 20,000 |
| `mark_medication_taken` | medication_name, person | time (si ambiguedad) | 15,000 | 15,000 |
| `list_medications` | person | date (default: hoy) | 10,000 | 10,000 |
| `check_medication_status` | person | date (default: hoy) | 10,000 | 5,802 |
| `add_reminder` | reminder_title, date, time, person | description | 20,000 | 20,000 |
| `list_reminders` | — | person, date (default: todo/hoy) | 8,000 | 8,000 |
| `add_contact` | contact_name, phone, person | — | 15,000 | 15,000 |
| `add_shopping_item` | product | quantity, store | 15,000 | 15,000 |
| `mark_shopping_done` | product | — | 10,000 | 8,096 |
| `list_shopping` | — | — | 5,000 | 460 |
| `greeting` | — | — | 3,000 | 607 |
| `help` | — | — | 3,000 | 480 |

**Total generado: 118,445 ejemplos** (intents simples sin entidades tienen menos variaciones posibles)

### Split del dataset
- Train: 94,756 ejemplos (80%)
- Validation: 11,844 ejemplos (10%)
- Test: 11,845 ejemplos (10%)

### Entidades

| Entidad | Ejemplos | Usado en |
|---------|----------|----------|
| `person` | "mi abuela", "Juan", "la yaya", "mamá" | medication, reminder, contact |
| `medication_name` | "ibuprofeno", "la pastilla azul", "omeprazol" | medication |
| `dose` | "500mg", "una pastilla", "medio comprimido" | add_medication |
| `frequency` | "cada 8 horas", "todos los dias", "lunes y miercoles" | add_medication |
| `time` | "a las 9", "por la mañana", "despues de comer" | medication, reminder |
| `date` | "mañana", "el jueves", "el 20 de marzo" | reminder, list |
| `reminder_title` | "cita medico", "analitica", "revision" | add_reminder |
| `description` | "en el hospital central", "llevar informes" | add_reminder |
| `contact_name` | "Dr. Garcia", "enfermera Maria" | add_contact |
| `phone` | "612345678" | add_contact |
| `product` | "tomates", "leche", "esparragos" | shopping |
| `quantity` | "2 kilos", "una docena", "3" | add_shopping_item |
| `store` | "Mercadona", "Lidl", "la farmacia" | add_shopping_item |

### Resolucion de entidad `person`
El modelo NER detecta que hay una referencia a persona. Luego la logica de negocio resuelve:
- Nombre directo → buscar en trackings del hogar
- Relacion ("mi abuela") → matchear con campo `relationship` del tracking
- Si hay ambiguedad → Claude API pregunta "¿Te refieres a Maria o a Carmen?"

---

## Modelos a comparar (5)

| Modelo | Tipo | Libreria | GPU necesaria |
|--------|------|----------|---------------|
| **SVM + TF-IDF** | Clasico ML | scikit-learn | No |
| **Random Forest + TF-IDF** | Ensamble clasico | scikit-learn | No |
| **Naive Bayes + TF-IDF** | Probabilistico | scikit-learn | No |
| **BiLSTM** | Red neuronal recurrente | PyTorch | Recomendada |
| **BERT multilingual fine-tuned** | Transformer | HuggingFace + PyTorch | Recomendada |

### Metricas de evaluacion
- Accuracy
- F1-score (macro y weighted, por intent)
- Precision y Recall por intent
- Matriz de confusion
- Latencia de inferencia (ms por ejemplo)
- Tiempo de entrenamiento

### Hardware para entrenamiento
- PC Windows con i9, 64GB RAM, RTX 4060
- Los 3 modelos clasicos (SVM, RF, NB) entrenan en segundos con CPU
- BiLSTM: unos minutos con GPU
- BERT: 10-20 minutos con GPU

---

## Estructura de archivos

```
tfm/
├── TFM_PROGRESS.md           # Este archivo
├── requirements.txt           # Dependencias Python
├── build_dataset.py           # Genera dataset con templates (local, sin API)
├── generate_dataset.py        # Generador alternativo con Claude API (no usado)
├── .env                       # API key (no se sube a git)
├── dataset/                   # Dataset generado (118K ejemplos)
│   ├── train.jsonl            # 94,756 ejemplos (80%)
│   ├── val.jsonl              # 11,844 ejemplos (10%)
│   ├── test.jsonl             # 11,845 ejemplos (10%)
│   └── {intent}.jsonl         # Archivos por intent individual
├── train_svm.py               # Entrenamiento SVM + TF-IDF
├── train_random_forest.py     # Entrenamiento Random Forest + TF-IDF
├── train_naive_bayes.py       # Entrenamiento Naive Bayes + TF-IDF
├── train_bilstm.py            # Entrenamiento BiLSTM (GPU)
├── train_bert.py              # Entrenamiento BERT multilingual (GPU)
├── compare_models.py          # Comparacion final + graficos
├── models/                    # Modelos entrenados (se genera al entrenar)
│   ├── svm_model.pkl
│   ├── random_forest_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── bilstm_model.pt
│   └── bert_model/
└── results/                   # Resultados (se genera al entrenar)
    ├── svm_results.json
    ├── random_forest_results.json
    ├── naive_bayes_results.json
    ├── bilstm_results.json
    ├── bert_results.json
    ├── comparison_summary.json
    └── plots/                 # Graficos comparativos
```

---

## Instrucciones de ejecucion

### En el PC Windows (i9 + RTX 4060)

```bash
# 1. Instalar PyTorch con CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Instalar el resto de dependencias
pip install -r requirements.txt

# 3. Generar dataset (si no esta generado)
python build_dataset.py

# 4. Entrenar cada modelo
python train_svm.py
python train_random_forest.py
python train_naive_bayes.py
python train_bilstm.py
python train_bert.py

# 5. Comparar resultados y generar graficos
python compare_models.py
```

### Copiar resultados al Mac
Copiar las carpetas `results/` y `models/` de vuelta al Mac para continuar con la implementacion del endpoint.

---

## Progreso

### 2026-03-15 — Definicion del proyecto
- [x] Definidos 12 intents con campos obligatorios y opcionales
- [x] Definidas 13 entidades
- [x] Decidida arquitectura hibrida (modelo propio + Claude API)
- [x] Decidido idioma: castellano (catalan como trabajo futuro)

### 2026-03-15 — Dataset
- [x] Creado script `build_dataset.py` con templates + combinaciones aleatorias
- [x] Generados 118,445 ejemplos (objetivo era 134K, intents simples tienen menos variaciones)
- [x] Split automatico: train 80% / val 10% / test 10%
- [x] Formato JSONL (estandar para NLP)
- [x] Se intento primero con Claude API (~$80 estimado) → descartado por coste
- [x] Generacion local con templates: gratis, instantaneo, reanudable

### 2026-03-15 — Scripts de entrenamiento
- [x] Creado `train_svm.py` — SVM + TF-IDF
- [x] Creado `train_random_forest.py` — Random Forest + TF-IDF
- [x] Creado `train_naive_bayes.py` — Naive Bayes + TF-IDF
- [x] Creado `train_bilstm.py` — BiLSTM con PyTorch (GPU)
- [x] Creado `train_bert.py` — BERT multilingual fine-tuned (GPU)
- [x] Creado `compare_models.py` — Comparacion con graficos
- [x] Fix encoding UTF-8 en scripts (emojis no compatibles con cp1252 de Windows)
- [x] Fix `total_mem` → `total_memory` en `train_bert.py` (API PyTorch)
- [x] Entrenado SVM — **99.49% accuracy**, 4.94s entrenamiento
- [x] Entrenado Random Forest — **99.42% accuracy**, 6.81s entrenamiento
- [x] Entrenado Naive Bayes — **99.39% accuracy**, 0.04s entrenamiento
- [x] Entrenado BiLSTM — **99.55% accuracy**, 131s entrenamiento (15 epochs, mejor: epoch 6)
- [x] Entrenado BERT — **99.57% accuracy**, 5166s entrenamiento (5 epochs, mejor: epoch 2)
- [x] Comparar resultados y elegir mejor modelo
- [x] Resultado comparacion: BERT 99.57% > BiLSTM 99.55% > SVM 99.49% > RF 99.42% > NB 99.39%
- [x] Modelo elegido: **BiLSTM** (0.02% menos que BERT pero 8x mas rapido, 70x menos tamaño)

### 2026-03-15 — Conversión y despliegue del modelo
- [x] Creado `convert_to_onnx.py` — convierte BiLSTM de PyTorch a ONNX
- [x] Modelo convertido: bilstm_model.onnx (9.7MB) + bilstm_vocab.json (22KB)
- [x] Copiados archivos ONNX a `functions/chatbot/`
- [x] Añadido `onnxruntime-node` como dependencia de Cloud Functions

### 2026-03-15 — Cloud Function chatbot
- [x] Creada Cloud Function callable `chatbot` en `functions/index.js`
- [x] Clasificacion de intent con BiLSTM (ONNX)
- [x] Extraccion de entidades con Claude API (Haiku — barato y rapido)
- [x] Resolucion de persona a tracking real del hogar
- [x] Dialogo multi-turno: si faltan campos obligatorios, pregunta uno a uno
- [x] Si confianza < 0.7, Claude API interpreta la frase
- [x] Acciones implementadas:
  - add_medication → crea medicamento en tracking
  - mark_medication_taken → marca toma como tomada
  - list_medications → lista medicamentos activos
  - check_medication_status → comprueba tomas del dia
  - add_reminder → crea recordatorio en tracking
  - list_reminders → lista recordatorios
  - add_contact → añade contacto a tracking
  - add_shopping_item → añade producto a la lista
  - mark_shopping_done → marca producto como comprado
  - list_shopping → muestra lista pendiente
  - greeting → respuesta de bienvenida
  - help → lista de funcionalidades

### 2026-03-15 — Logging y monitorización del chatbot
- [x] Añadido logging completo a la Cloud Function chatbot
- [x] Cada interaccion se guarda en `admin/chatbotLogs/entries/` con: mensaje, intent, confianza, entidades, llamadas a Claude API, accion ejecutada, errores, duracion
- [x] Logs detallados en Firebase Console con emojis paso a paso

### 2026-03-15 — Chat UI en Flutter
- [x] Creado `lib/services/chatbot_service.dart` — llama a la Cloud Function y gestiona contexto de conversacion
- [x] Creado `lib/screens/home/dashboard/chatbot_screen.dart` — pantalla de chat con burbujas, indicador de "escribiendo", boton limpiar, badge "accion ejecutada"
- [x] Boton "Asistente IA" añadido al Dashboard para todos los usuarios
- [x] Navegacion `/chatbot` en dashboard_screen.dart

### 2026-03-15 — Testing y bugfixes
- [x] Deploy y primer test del chatbot
- [x] **Bug fix**: "que puedes hacer?" clasificado como greeting (48.6%) → añadido sistema de overrides por keywords para forzar intent correcto
- [x] **Bug fix**: "añade un medicamento" clasificado como add_shopping_item → override si contiene "medicamento/pastilla/medicina"
- [x] **Bug fix**: resolvePerson matcheaba trackings con nombre vacio (linked users) → ahora busca nombre en `users/{linkedUserId}` para trackings con app
- [x] **Bug fix**: add_shopping_item no guardaba en Firestore → cambiado `set` con merge por `update` con verificacion previa
- [x] **Bug fix**: "para mi/yo/a mí" no resolvia a persona → añadida deteccion de auto-referencia, busca tracking con `linkedUserId === userId`
- [x] **Bug fix**: "demo2" matcheaba "demo" por substring → cambiado a match por palabra completa
- [x] **Bug fix**: usuario cambia de tema durante follow-up (ej: "Marca medicamento..." mientras preguntaba dosis) → detecta frases de 4+ palabras, clasifica con BiLSTM, si es intent diferente con >85% confianza abandona follow-up y procesa como nueva conversacion
- [x] Reordenado: check de confianza baja ANTES de greeting/help para que frases ambiguas deleguen a Claude API

### 2026-03-16 — Segundo round de testing y bugfixes
- [x] Añadido logging en consola Flutter (chatbot_service.dart) — muestra usuario/bot/contexto/tiempo en debug console
- [x] **Bug fix**: shopping items con multiples productos ("leche, pan y huevos") → Claude devolvia array en `product`, rompia la lista. Ahora separa en items individuales (soporta arrays de Claude y strings con comas/"y")
- [x] **Bug fix**: shopping items se escribian con formato incorrecto en Firestore → igualado al formato exacto de ShoppingService.addItem() de Flutter
- [x] **Bug fix**: mark_medication_taken usaba hora actual como takeId (ej: `20260316_0015`) pero la app busca por horario programado (ej: `20260316_0900`) → ahora busca los schedules del medicamento, encuentra la primera toma pendiente del dia, y usa ese takeId. Si ya estan todas tomadas devuelve "ya esta tomado hoy"
- [x] **Bug fix**: "Ya me he tomado el paracetamol de hoy" se clasificaba como mark_shopping_done → añadido router previo por reglas y override para forzar mark_medication_taken en frases claras de medicacion tomada

### Tests exitosos (2026-03-16)
- [x] "Añade leche, pan y huevos a la lista" → 3 productos añadidos correctamente
- [x] "Compra arroz, pasta y aceite" → 3 productos añadidos
- [x] "¿Qué hay en la lista de la compra?" → lista correcta de 6 items
- [x] "ya he comprado pan" → marcado como comprado
- [x] "Marca como tomado el paracetamol" → pregunta para quien → "Demo" → marcado
- [x] "Marca como tomado el paracetamol de Demo" → ejecutado directamente
- [x] "que medicamentos toma Demo?" → "Paracetamol (1)"
- [x] "que medicamentos toma Francisco?" → resuelto a Francisco Ballarin

### 2026-05-21 — Robustez del chatbot y control de ruido
- [x] Creado `tfm/evaluation/chatbot_intent_cases.jsonl` con 26 frases reales/manuales para medir errores de intent (ruido, frases sin tildes, compra vs medicacion, preguntas vs acciones, ayuda, saludos, recordatorios y contactos)
- [x] Creado `tfm/evaluation/evaluate_chatbot_router.js` para validar localmente las reglas sin Firebase, Claude API ni ONNX
- [x] Extraida logica pura a `functions/chatbot/router.js`
  - normalizacion para reglas: minusculas, sin tildes, limpieza de puntuacion, muletillas y variantes frecuentes de STT/escritura
  - normalizacion separada para el modelo: limpia puntuacion pero conserva tildes para no romper el vocabulario entrenado
  - router determinista previo al BiLSTM para casos de alta precision
- [x] Nuevas reglas conservadoras para:
  - `mark_medication_taken`: "me he tomado", "ha tomado", "marca como tomado", etc. + pista de medicamento
  - `check_medication_status`: "ha tomado todo", "queda alguna pastilla pendiente", "comprueba la medicacion"
  - regla defensiva: una pregunta tipo "¿ha tomado el paracetamol?" no marca nada como tomado; se trata como consulta de estado
  - `mark_shopping_done`: "he comprado", "marca como comprado"
  - `add_shopping_item`, `list_shopping`, `list_medications`, `add_medication`, `help`, `greeting`
- [x] El flujo ahora usa reglas primero; si no hay match claro, usa BiLSTM
- [x] Añadido override post-modelo para corregir confusiones conocidas entre medicacion y compra
- [x] Subido control de ejecucion: si solo el modelo propone una accion con confianza media (<0.85), el bot no ejecuta y pide reformular
- [x] Ampliado logging de `admin/chatbotLogs/entries/` con texto normalizado, fuente de decision (`rules`, `model`, `override`), razon de routing, intent/confianza del modelo y alternativas top-3
- [x] Mejora UX: el input del chat mantiene el foco despues de enviar un mensaje o limpiar la conversacion, evitando tener que pulsar el campo cada vez
- [x] Validacion local:
  - `node tfm/evaluation/evaluate_chatbot_router.js` → 26 casos, 24 cubiertos por reglas, 2 delegados al modelo, 0 errores de reglas
  - `node -c functions/index.js`
  - `node -c functions/chatbot/router.js`
  - `eslint chatbot/router.js` desde `functions/`
- [ ] Pendiente: test manual con app/Firebase para comprobar entidades, follow-ups y escritura real en Firestore
- [ ] Pendiente: valorar confirmacion explicita antes de ejecutar acciones sensibles de medicacion

### Pendiente
- [ ] Más testing del chatbot con casos reales
- [ ] STT (speech-to-text) con `speech_to_text` de Flutter
- [ ] TTS (text-to-speech) con `flutter_tts`
- [ ] Pruebas con usuarios reales

---

## Decisiones tomadas

| Decision | Razon |
|----------|-------|
| Castellano solo (de momento) | Simplificar, catalan como trabajo futuro |
| Hibrido modelo propio + Claude API | Mejor resultado y mas chicha academica |
| Dataset generado localmente con templates | Claude API costaba ~$80, templates son gratis e instantaneos |
| 5 modelos en vez de 3 | Mas comparaciones = mejor TFM (SVM, RF, NB, BiLSTM, BERT) |
| Volumen variable por intent | Intents complejos necesitan mas variaciones, simples tienen limite natural |
| BERT multilingual como candidato | Facilita añadir catalan despues sin reentrenar desde cero |
| Entrenar en PC Windows con GPU | RTX 4060 acelera BERT de horas a minutos |
| JSONL como formato | Estandar NLP, carga directa en HuggingFace datasets |
