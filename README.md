# Asistente conversacional para una app de cuidados

Trabajo de fin de máster (Big Data e IA). Proyecto personal, desarrollador único, construido con
ayuda de IA (Claude Code y Codex) como parte de mi flujo de trabajo diario; la metodología, el
análisis que sigue y las conclusiones son míos.

### Qué hace

Plamily es una app que construí para familias que coordinan el cuidado de un familiar mayor:
medicación, recordatorios, lista de la compra compartida, contactos. Este asistente convierte una
frase —escrita o dictada— en una acción dentro de la app: añadir un medicamento, marcar una toma
como tomada, crear un recordatorio o apuntar algo en la compra.

En lugar de recorrer cuatro pantallas para apuntar una pastilla, dices *«apúntame paracetamol cada
8 horas»* y queda hecho. El objetivo a medio plazo es exponerlo a través de Alexa o Siri, de forma
que el cuidador no tenga ni que abrir la app: la mayoría de estas anotaciones se hacen con las
manos ocupadas.

### Qué estudia el TFM

La primera etapa de ese pipeline: **decidir cuál de las 12 acciones posibles está pidiendo el
usuario**. Cinco familias de modelos comparadas sobre 118.445 frases en español; el BiLSTM ganador
se exportó a ONNX y se sirve desde una Cloud Function.

### Qué encontré

Lo interesante de este repositorio no es la tabla de accuracy: es la auditoría que explica por qué
esa tabla no significa casi nada.

El pipeline de entrenamiento, los ficheros de resultados en bruto y los artefactos de despliegue
están todos aquí. La app para la que se construyeron es privada; las rutas bajo `functions/`
apuntan a ese código privado.

## TL;DR

- Naive Bayes, Random Forest, LinearSVC, un BiLSTM y BERT multilingüe se quedan todos entre el
  **99,39% y el 99,57%** de accuracy en test. Veintiún errores separan al mejor del peor, sobre
  11.845 ejemplos de test. Sus intervalos de Wilson al 95% se solapan por completo, así que el
  ranking no es defendible.
- Después medí qué contiene realmente el conjunto de test: **el 98,66% de los ejemplos de test se
  generaron a partir de una plantilla que también aparece en train** y **4 de 103.574 tokens
  de test** quedan fuera del vocabulario. La cifra de portada mide memorización de plantillas, no
  generalización.
- **52 de los 53 errores del BiLSTM** caen sobre una misma cadena de plantilla que mi propio
  generador emite con dos etiquetas distintas (`build_dataset.py:407` y `build_dataset.py:582`). El
  error residual es un bug del dataset, no un límite del modelado.
- El BiLSTM se exportó a ONNX y se sirve desde una Cloud Function. Ese despliegue tiene un
  **desajuste de tokenización entre entrenamiento y servicio** real y reproducible, medido en la
  sección 7. **No está integrado en la app publicada**.

---

## 1. La tarea

Plamily es una app Flutter que construí para familias que coordinan el cuidado de un familiar
mayor: pautas de medicación, recordatorios, una lista de la compra compartida, contactos. El TFM se
pregunta si un punto de entrada por texto (y más adelante por voz) puede reducir un flujo de varias
pantallas a una sola frase: *«ya me he tomado el paracetamol»*.

El pipeline se divide en tres etapas y solo la primera es objeto del TFM:

| Etapa | Cómo se resuelve | ¿Está en este repo? |
|---|---|---|
| Clasificación de intenciones | Modelo entrenado, 12 clases | Sí — toda la comparación |
| Extracción de entidades | Llamada a la API de Claude Haiku (`functions/index.js:3565`) | Desplegada, sin medir |
| Ejecución de la acción | Escrituras en Firestore, una por intención (`functions/index.js:3742`) | Desplegada, sin medir |

Esto importa para leer los resultados con honestidad: **el modelo entrenado solo hace clasificación
de intenciones**. La extracción de entidades se delega en un LLM en tiempo de petición. El plan
original incluía una cabeza de NER; no llegó a construirse.

### Las 12 intenciones

`build_dataset.py` pide a cada generador un número objetivo de ejemplos y se detiene antes de tiempo
cuando la deduplicación dentro de la intención agota el espacio de plantillas. Cinco intenciones
tocan ese techo, y de ahí viene el desbalance de clases:

| Intención | Objetivo | Generados | Soporte en test | Qué hace en la app |
|---|---:|---:|---:|---|
| `add_medication` | 20,000 | 20,000 | 2,038 | Crea un medicamento con dosis, frecuencia y horario |
| `add_reminder` | 20,000 | 20,000 | 2,019 | Crea un recordatorio con fecha para una persona bajo seguimiento |
| `mark_medication_taken` | 15,000 | 15,000 | 1,446 | Marca como tomada la dosis pendiente del día |
| `add_contact` | 15,000 | 15,000 | 1,506 | Añade un contacto telefónico a una persona bajo seguimiento |
| `add_shopping_item` | 15,000 | 15,000 | 1,529 | Añade artículos a la lista de la compra compartida |
| `list_medications` | 10,000 | 10,000 | 1,014 | Lista los medicamentos activos de una persona |
| `check_medication_status` | 10,000 | **5,802** | 565 | Informa de qué dosis siguen pendientes hoy |
| `mark_shopping_done` | 10,000 | **8,096** | 791 | Marca un artículo de la compra como comprado |
| `list_reminders` | 8,000 | 8,000 | 791 | Lista los próximos recordatorios |
| `list_shopping` | 5,000 | **460** | 45 | Muestra la lista de la compra pendiente |
| `greeting` | 3,000 | **607** | 44 | Respuesta enlatada directa |
| `help` | 3,000 | **480** | 57 | Respuesta enlatada directa |
| **Total** | 134,000 | **118,445** | 11,845 | |

El desbalance de clases es de **43,5x** (`add_medication` 20.000 frente a `list_shopping` 460).
Random Forest y LinearSVC usan `class_weight="balanced"` (`train_random_forest.py:80`,
`train_svm.py:82`); Naive Bayes y los dos modelos neuronales no compensan nada. Por eso **la métrica
que hay que leer en la tabla siguiente es la F1-macro, no la accuracy**.

---

## 2. El dataset es 100% sintético

No hay forma de esquivarlo, y va primero porque es la primera pregunta que debería hacer cualquiera
que revise esto.

**Por qué sintético.** La app no tenía ninguna superficie conversacional, así que no había ningún
registro de frases reales de usuario que explotar: un arranque en frío con cero datos del dominio.
Había dos opciones sobre la mesa:

1. Generar con un LLM (`generate_dataset.py`, `claude-sonnet-4-20250514` en la línea 326). Coste
   estimado de unos 80 $ para el volumen objetivo. **Nunca se ejecutó**: descartado por coste
   (`TFM_PROGRESS.md:196`).
2. Generar en local a partir de plantillas fijas más huecos de entidad aleatorizados
   (`build_dataset.py`, 969 líneas). Gratis, instantáneo, determinista. **Esto es lo que produjo los
   datos.**

**Cómo funciona.** Doce funciones generadoras componen plantillas f-string con bolsas de entidades
curadas: 73 referencias a personas (`"mi abuela"`, `"la yaya"`, `"mamá"`, nombres propios),
67 nombres de medicamento incluyendo los coloquiales (`"la pastilla de la tensión"`), 90 productos,
25 tiendas, 41 horas, 34 fechas. Una pasada de corrupción ortográfica y de erratas con un 8% de
probabilidad (`build_dataset.py:217`) simula las costumbres de la escritura en móvil en español
—comerse las tildes y abreviar— (`"qué"` → `"que"`, `"por favor"` → `"xfa"`).

Si se reconstruye el esqueleto de cada ejemplo (sustituyendo los valores de entidad de vuelta por
sus marcadores) salen **2.424 pares `(intent, skeleton)` distintos** para 118.445 ejemplos: una
media de 49 ejemplos por esqueleto. Si nos limitamos a las nueve intenciones que llevan entidades,
son 1.337 esqueletos para 117.358 ejemplos, es decir, 88 ejemplos por esqueleto. La sección 4.2 es
la consecuencia directa.

**Reproducibilidad.** `random.seed(42)` en `build_dataset.py:14`, sin llamadas a API y sin estado
externo. Reejecuté el generador desde un checkout limpio mientras escribía este README y obtuve el
mismo split, ejemplo por ejemplo, en 1,5 s:

```
TOTAL: 118,445 examples generated
  train: 94,756   val: 11,844   test: 11,845
```

**Lo que estos datos no son.** No son habla real de usuarios. El vocabulario, la longitud de las
frases y la variedad sintáctica están limitados por lo que se me ocurrió al escribir las plantillas.
Todas las afirmaciones de este repositorio están limitadas por ese mismo techo.

---

## 3. Los cinco modelos

Entrenados en una workstation i9 / 64 GB / RTX 4060 (`TFM_PROGRESS.md:108`). Los modelos clásicos en
CPU, los neuronales en GPU.

| Modelo | Configuración | Accuracy | F1-macro | Errores /11,845 | Params | Tiempo de entrenamiento |
|---|---|---:|---:|---:|---:|---:|
| Naive Bayes + TF-IDF | `MultinomialNB(alpha=0.1)`, n-gramas 1-2, 50k features | 0.9939 | 0.9890 | 72 | — | **0.036 s** |
| Random Forest + TF-IDF | 200 árboles, balanced | 0.9942 | 0.9896 | 69 | — | 6.81 s |
| SVM (LinearSVC) + TF-IDF | `C=1.0`, `max_iter=10000`, balanced | 0.9949 | 0.9912 | 60 | — | 4.94 s |
| **BiLSTM** | 2x256 ocultas, emb. 128 dim., dropout 0.3, 15 épocas (mejor: 6) | 0.9955 | **0.9941** | 53 | 2,550,796 | 131 s |
| BERT multilingual | `bert-base-multilingual-cased`, lr 2e-5, 3 épocas (mejor: 2) | **0.9957** | 0.9937 | 51 | 177,862,668 | 3,276 s |

Los tres vectorizadores TF-IDF conservan las tildes (`strip_accents=None`): una decisión explícita
por ser español. El vocabulario del BiLSTM se construye solo con train, con `MIN_WORD_FREQ = 2`,
tokenizando con `text.lower().split()` (`train_bilstm.py:77-88`).

Por intención, el BiLSTM saca **F1 = 1,0000 en nueve de las doce clases**. Un resultado perfecto en
nueve clases no es un logro: es un síntoma, y es lo que me hizo empezar a mirar.

---

## 4. Por qué estos números no significan lo que parece

Esta es la sección de la que va realmente el TFM.

### 4.1 El ranking no tiene contenido estadístico

La diferencia entre el mejor y el peor es de **21 errores sobre 11.845**. Intervalos de Wilson al
95%:

| Modelo | Accuracy | IC Wilson 95% |
|---|---:|---|
| Naive Bayes | 99.39% | [99.24, 99.52] |
| Random Forest | 99.42% | [99.26, 99.54] |
| SVM | 99.49% | [99.35, 99.61] |
| BiLSTM | 99.55% | [99.42, 99.66] |
| BERT | 99.57% | [99.43, 99.67] |

Todos los intervalos se solapan entre sí. La cota superior de Naive Bayes (99,52) queda por encima
de la cota inferior de BERT (99,43). **«BERT es el mejor modelo» no es una afirmación que este
experimento pueda sostener.** Cualquier TFM que presente ese orden como un hallazgo está leyendo en
sus propios datos más de lo que hay.

### 4.2 El 98,66% del conjunto de test se filtra desde train

El generador produce primero los 118.445 ejemplos y después aplica un `random.shuffle()` plano antes
de cortar 80/10/10 (`build_dataset.py:945-955`). Las plantillas no se reservan, y con una media de
49 ejemplos por esqueleto prácticamente todos los esqueletos acaban en los tres lados del split.

Medido reconstruyendo el esqueleto de cada ejemplo e intersecando los conjuntos:

```
distinct (intent, skeleton) pairs in train       : 2,105
test examples whose skeleton is already in train : 11,686 / 11,845 = 98.66%
```

Es una fuga de formulación, no de filas: la deduplicación es por cadena exacta dentro de cada
intención, así que solo **16 de 11.845** ejemplos de test aparecen literalmente en train. El modelo
no ha memorizado las filas de test; ha memorizado los 2.105 patrones de frase de los que
salieron.

De los 159 ejemplos de test cuyo esqueleto es genuinamente nuevo, 146 pertenecen a las tres
intenciones sin huecos de entidad (`help` 57, `list_shopping` 45, `greeting` 44), donde esqueleto y
texto son la misma cadena por construcción. Quedan **13 ejemplos de 11.845** repartidos entre las
nueve intenciones con entidades que presentan una formulación que el modelo no ha visto nunca.

### 4.3 El vocabulario también se filtra

```
test tokens                                   : 103,574
tokens unseen in train                        :       4  (0.004%)
test examples containing zero unseen tokens   : 11,841 / 11,845  (99.97%)
```

Los cuatro tokens no vistos son `dias!`, `dia?`, `ola!`, `saludos!!`: artefactos de puntuación, no
palabras nuevas. (Usando el vocabulario real del modelo, que descarta los hápax con
`MIN_WORD_FREQ = 2`, la cuenta sube a 7 de 103.574; la conclusión es la misma.)

**Juntas, 4.2 y 4.3 dicen que el conjunto de test es una prueba de memorización.** Un modelo que se
limite a guardar pares esqueleto→etiqueta y nada más saca aquí en torno al 98,7%. Ese es el suelo
justo por encima del cual están apiñados estos cinco modelos, que es exactamente lo que parece la
horquilla de 21 errores.

### 4.4 La mayor parte del error restante es ruido de etiqueta que introduje yo

La deduplicación de `generate_intent()` usa un conjunto `seen_texts` que es **local a cada
intención** (`build_dataset.py:883`). Nada comprueba si una misma cadena se emite con dos etiquetas
distintas. Medido sobre el dataset completo:

```
texts appearing with contradictory labels : 87
  (list_medications, list_reminders)      : 86
  (greeting, help)                        :  1
```

Los 86 vienen de una única plantilla —`"¿Qué le toca a {person} {date}?"`— escrita en
`gen_list_medications` en **`build_dataset.py:407`** y, literalmente igual, en `gen_list_reminders`
en **`build_dataset.py:582`**. La frase es genuinamente ambigua en español (¿qué le toca: la
medicación o una cita?), y yo la etiqueté de las dos maneras. En el dataset completo ese esqueleto
lleva la etiqueta `list_medications` 501 veces y `list_reminders` 476 veces.

La matriz de confusión del BiLSTM tiene exactamente dos celdas fuera de la diagonal:

```
list_medications -> list_reminders : 52
greeting         -> list_reminders :  1
```

| Intención | Precisión | Recall | Soporte |
|---|---:|---:|---:|
| `list_medications` | 1.0000 | 0.9487 | 1,014 |
| `list_reminders` | 0.9372 | 1.0000 | 791 |
| `greeting` | 1.0000 | 0.9773 | 44 |

El split de test contiene **exactamente 52** ejemplos de `list_medications` construidos a partir del
esqueleto en colisión, y el modelo comete **exactamente 52** errores de `list_medications`, todos
predichos como `list_reminders`. La matriz de confusión por sí sola no identifica los ejemplos uno a
uno, pero los recuentos coinciden y el mecanismo se ve en el código del generador. **52 de los 53
errores del modelo caen sobre ruido de etiqueta que introduje yo mismo**, en una frase que no tiene
respuesta correcta. El 53º es un único ejemplo de `greeting`, también predicho como
`list_reminders`, que la colisión no explica.

El BiLSTM no acierta el 99,55% porque sea bueno. Acierta el 99,55% porque el 98,66% del conjunto de
test es memorizable y la mayor parte del 1,34% restante es una contradicción de etiquetado.

### 4.5 Las clases minoritarias apenas tienen soporte en test

`greeting` tiene 44 ejemplos en test, `list_shopping` 45 y `help` 57. Un solo error mueve el recall
de `greeting` 2,3 puntos. La F1-macro —la métrica que defiendo como la correcta con un desbalance de
43,5x— queda por tanto dominada por tres clases cuya medición es muy ruidosa. Esa es una segunda
razón para no leer el ranking macro como un orden fino.

### 4.6 Qué haría distinto

1. **Split por grupos a nivel de plantilla.** Asignar cada `(intent, skeleton)` a exactamente uno de
   train/val/test *antes* de rellenar los huecos de entidad, usando `GroupShuffleSplit` sobre el id
   de esqueleto. Así la accuracy de test mide la generalización a formulaciones no vistas, que es la
   magnitud que la app necesita de verdad.
2. **Deduplicación global con detección de conflictos.** Sacar `seen_texts` de `generate_intent()`,
   hacerlo global al dataset y abortar con error ante cualquier cadena emitida con dos etiquetas.
   Eso saca a la luz la colisión de `"¿Qué le toca a...?"` en el momento de generar, en vez de en la
   matriz de confusión.
3. **Fusionar o desambiguar las intenciones que colisionan.** `list_medications` y `list_reminders`
   necesitan o bien una intención común de «qué hay programado» con un desambiguador aguas abajo, o
   bien plantillas que sean realmente distinguibles.
4. **Reportar el test de McNemar** entre pares de modelos en lugar de un orden bruto por accuracy.

**Mi expectativa —sin comprobar, y etiquetada como tal—:** con un split por grupos a nivel de
plantilla espero que los cinco modelos caigan de forma apreciable, y espero que la distancia entre
ellos se abra, con los modelos basados en embeddings (BiLSTM, mBERT) degradándose menos que los
n-gramas TF-IDF ante formulaciones no vistas, porque una bolsa de bigramas no vistos no lleva
señal. No he ejecutado este experimento, así que esto es una hipótesis, no un resultado.

---

## 5. Lo que la comparación sí demuestra

Quitando el ranking, dos conclusiones sobreviven a la auditoría.

**Un transformer de 178M de parámetros no le gana en esta tarea a un Naive Bayes de 36
milisegundos.** BERT cuesta **3.276 s de entrenamiento en GPU (55 min) y 69,7x más parámetros** que
el BiLSTM para comprar **0,017 puntos porcentuales** de accuracy —dos errores sobre 11.845, muy
dentro del ruido— y encima *pierde* en F1-macro. Frente a Naive Bayes, compra 21 errores a cambio de
multiplicar por 92.000 el tiempo de entrenamiento. Para un vocabulario de intenciones acotado y de
conjunto cerrado, la restricción que manda no es la capacidad del modelo: son los datos.

**El BiLSTM es la elección correcta con desbalance.** Gana en F1-macro (0,9941 frente al 0,9937 de
BERT) siendo 25x más rápido de entrenar y 69,7x más pequeño. Con un desbalance de clases de 43,5x,
la F1 promediada por macro es la métrica que se niega a dejar que las tres intenciones grandes tapen
a las pequeñas. El BiLSTM se eligió para el despliegue por eso y por el tamaño del artefacto: 9,7
MiB de ONNX caben dentro del bundle de la función, donde un modelo de 178M de parámetros no cabría.

**Salvedad, dicha sin rodeos.** Los argumentos de tamaño y F1-macro son sólidos; el argumento de
*latencia* del TFM no está bien medido. `avg_latency_ms` se calcula como el tiempo de reloj de la
inferencia sobre el batch completo dividido entre N (`train_svm.py:99`, `train_bilstm.py:265`), es
decir, throughput amortizado, no latencia por petición en un contenedor serverless. Por eso parece
que el SVM tarda 0,0002 ms por ejemplo. **No hay ningún benchmark de la sesión ONNX desplegada ni
ninguna medida de cold start.** El experimento correcto es la latencia por petición de
`onnxruntime-node` en una instancia de 512 MiB, en caliente y en frío. No se ejecutó, así que la
afirmación de que un artefacto más pequeño ayuda en el cold start es un razonamiento, no un dato.

---

## 6. Despliegue

### Exportación

```python
torch.onnx.export(
    model, dummy_input, "models/bilstm_model.onnx",
    input_names=["input_ids"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "logits": {0: "batch_size"}},
    opset_version=14,
)
```
`convert_to_onnx.py:82-93`

Resultado: `input_ids` INT64 `[batch, 50]` → `logits` FLOAT `[batch, 12]`, un grafo de 39 nodos.
**10.207.804 bytes (9,7 MiB)** de ONNX más un JSON de vocabulario de **22 KB** (1.384 entradas
incluyendo `<PAD>`/`<UNK>`, 12 etiquetas, `max_seq_len: 50`). Los dos ficheros están commiteados
bajo `deployment/` y se leen con `fs.readFileSync` desde `__dirname`, así que la función no descarga
nada en el arranque en frío.

### Servicio

```js
exports.chatbot = onCall({
  region: "europe-west1",
  enforceAppCheck: true,
  secrets: [anthropicApiKey],
  memory: "512MiB",
  timeoutSeconds: 30,
}, async (request) => { ... });
```
`functions/index.js:4071-4076`

- `onnxruntime-node ^1.21.0` sobre Node 24, callable de Firebase Functions v2.
- La `InferenceSession` se guarda en una variable de ámbito de módulo y se crea de forma perezosa
  (`functions/index.js:3463-3471`), así que solo la primera invocación en un contenedor paga el
  montaje de la sesión.
- App Check obligatorio, más una comprobación explícita de `request.auth`. El `currentHomeId` de
  quien llama se lee una sola vez (`functions/index.js:4119`) y todas las lecturas y escrituras
  posteriores quedan acotadas bajo `homes/{homeId}/...`.
- El softmax se calcula en JS con la habitual resta del máximo para dar estabilidad numérica
  (`functions/index.js:3493-3496`), y las 3 alternativas principales se registran en Firestore.

### Enrutado: primero reglas, después modelo, después override

No se confía en el modelo para disparar escrituras por su cuenta. `deployment/router.js` (234
líneas) es un módulo puro, sin ningún `require` de Firebase, ONNX ni de la red, que contiene cuatro
umbrales de confianza (`router.js:3-8`):

| Umbral | Valor | Significado |
|---|---:|---|
| `rule` | 0.98 | Ha casado una regla determinista |
| `override` | 0.95 | Una regla ha corregido la salida del modelo |
| `modelDirect` | 0.85 | Por debajo de aquí, una intención decidida solo por el modelo **no** ejecuta ninguna acción |
| `lowFallback` | 0.70 | Por debajo de aquí, se le pasa la frase a Claude para que pida una aclaración |

Flujo (`functions/index.js:4220-4301`): primero se ejecuta `routeChatbotIntentByRules()`; solo si
devuelve `null` corre el modelo ONNX, seguido de `applyChatbotIntentOverride()`, que contiene cuatro
correcciones para pares de confusión conocidos (consulta de estado frente a marcar como tomado,
medicación frente a compra, ayuda frente a saludo). Una intención decidida solo por el modelo con
confianza entre 0,70 y 0,85 devuelve *«No estoy seguro de si quieres X. ¿Puedes decírmelo de otra
forma?»* en lugar de escribir en Firestore (`functions/index.js:4290-4300`). Para una app que
registra tomas de medicación, negarse a actuar ante una conjetura de confianza media es el
comportamiento por defecto correcto.

Una de las reglas es defensiva por diseño: `"¿ha tomado el paracetamol?"` es una **pregunta**, así
que se enruta a `check_medication_status` y nunca a `mark_medication_taken` (`router.js:123-127`,
aplicada en `router.js:181-183`). Un clasificador que marca una dosis como tomada porque el usuario
*preguntó* si se había tomado es un fallo de seguridad del paciente.

Las reglas cubren 9 de las 12 intenciones. `add_reminder`, `add_contact` y `list_reminders` no
tienen regla y siempre llegan al modelo.

### Banco de pruebas de evaluación

```js
const {
  routeChatbotIntentByRules,
} = require("../../deployment/router");
```
`tfm/evaluation/evaluate_chatbot_router.js:5-7`

El banco de pruebas importa **exactamente el mismo módulo que carga la función desplegada**, no una
reimplementación. Si las reglas cambian, el banco de pruebas cambia con ellas. 26 casos escritos a
mano (entrada sin tildes, formulación coloquial, ambigüedad compra-medicación, preguntas frente a
órdenes):

```
Casos totales: 26
Cubiertos por reglas: 24
Delegados al modelo: 2
Errores de reglas: 0
```

Los dos casos delegados son las frases de `add_reminder` y `add_contact`, que ninguna regla reclama.

**Limitaciones:** esos 26 casos los escribió la misma persona que escribió las reglas que ponen a
prueba, así que esto es una red de seguridad frente a regresiones, no una evaluación imparcial.
Además cubre solo la capa de reglas: nunca carga el modelo ONNX.

### Estado

La Cloud Function, el router, los artefactos ONNX, `lib/services/chatbot_service.dart` (81 líneas) y
una pantalla de chat en Flutter (`lib/screens/home/dashboard/chatbot_screen.dart`, 448 líneas) viven
todos en la rama `tfm`, y se probaron a mano de extremo a extremo contra un proyecto de Firebase
real (el registro de pruebas está en `tfm/TFM_PROGRESS.md:273-281`). **No están mergeados en
`master` y el chatbot no está presente en la build publicada de la app.** La sección 7 es el motivo
principal.

---

## 7. Problemas conocidos

### 7.1 Desajuste de tokenización train/serve (bug real, sin corregir)

El entrenamiento tokeniza con:

```python
def tokenize(text):
    return text.lower().split()
```
`train_bilstm.py:77-78`: minúsculas y separación por espacios en blanco. **Se conservan la
puntuación y las tildes**, así que el vocabulario aprendido contiene entradas como `¿cuántas`,
`dra.`, `mañana`, `medicación`.

Producción tokeniza con:

```js
function normalizeChatbotTextForModel(text) {
  return compactSpaces(String(text || "")
      .toLowerCase()
      .replace(/[`´'"]/g, " ")
      .replace(/[¿?¡!,.;()[\]{}]/g, " "));
}
```
`deployment/router.js:71-76`: **elimina la puntuación** y conserva las tildes.

Pasando esa función sobre todas las entradas del `bilstm_vocab.json` que se distribuye (1.382
entradas excluyendo `<PAD>`/`<UNK>`):

| Efecto | Recuento |
|---|---:|
| Entradas del vocabulario que el normalizador JS alteraría | 587 / 1,382 (42.5%) |
| Entradas que quedan así **inalcanzables** (la forma alterada no está en el vocabulario) | 45 |
| Entradas con tildes | 237 |
| …de las cuales no tienen gemelo sin tildes en el vocabulario | **221** |

Conviven dos modos de fallo. Primero, hay 45 entradas del vocabulario (`¿cuántas`, `dra.`, `2.5mg`,
…) que el tokenizador de producción no puede producir nunca: el modelo tiene pesos a los que ya no
puede llegar. Segundo, y peor en la práctica: los usuarios españoles escriben habitualmente sin
tildes, y el normalizador de *reglas* de este mismo repo elimina las tildes partiendo de esa premisa
(`router.js:43-45, 51-55`). El normalizador del *modelo* las conserva, y 221 entradas con tilde no
tienen equivalente sin tilde, de modo que `"medicacion"` escrito sin tilde cae en `<UNK>` mientras
que `"medicación"` resuelve. La pasada de erratas de los propios datos de entrenamiento
(`build_dataset.py:217`) genera a propósito variantes sin tildes, lo que enmascara esto en parte,
pero no lo cierra.

**No hay ningún test de paridad de tokenización Python↔JS.** Ese es el artefacto que falta. El
arreglo es una especificación de normalización compartida más un fichero de fixtures con, digamos,
200 cadenas cuyas secuencias de ids de token deban ser idénticas en los dos runtimes, ejecutado en
CI. Haber encontrado esto auditando mi propio despliegue es la razón de que el chatbot no se
mergeara.

### 7.2 La única confusión que comete el modelo no está protegida

Las reglas devuelven `null` para `"¿Qué le toca a mamá hoy?"` y para su forma sin tildes, y ningún
override cubre el par `list_medications` / `list_reminders`. La ambigüedad exacta que produce 52 de
los 53 errores del modelo le llega sin ninguna capa determinista delante ni ninguna corrección
detrás.

### 7.3 Latencia de servicio sin medir

Tratado en la sección 5: no hay benchmark de ONNX por petición ni medida de cold start. La decisión
de BiLSTM en lugar de BERT está bien respaldada por tamaño de artefacto y F1-macro, y sin respaldo
en latencia.

### 7.4 La extracción de entidades queda fuera del TFM

Toda acción necesita entidades, y toda entidad sale de una llamada a Claude Haiku
(`functions/index.js:3565-3607`). Eso significa una dependencia de una API externa en cada petición
dentro de un timeout de 30 s, un coste por interacción y un componente sin ninguna medida de
precisión de ningún tipo. Una cabeza de NER entrenada junto al clasificador —usando los spans de
entidad que el generador ya conoce— es el siguiente paso evidente y está sin hacer.

### 7.5 Un solo idioma, un solo registro

Solo español. El registro del dataset es peninsular y familiar-coloquial (`la yaya`, `Mercadona`,
`el estanco`). `bert-base-multilingual-cased` se eligió en parte para abaratar una extensión
posterior al catalán, pero no se ejecutó ninguna evaluación multilingüe.

---

## 8. Cómo reproducirlo

```bash
git checkout tfm
cd tfm

# 1. Dataset - deterministic, no API keys, ~1.5 s
python build_dataset.py          # -> dataset/{train,val,test}.jsonl

# 2. Classical models (CPU, seconds)
pip install -r requirements.txt
python train_naive_bayes.py
python train_random_forest.py
python train_svm.py

# 3. Neural models (GPU strongly recommended)
python train_bilstm.py           # 131 s on an RTX 4060
python train_bert.py             # 3,276 s on an RTX 4060

# 4. Comparison table and plots
python compare_models.py         # -> results/comparison_summary.json, results/plots/

# 5. Export the deployment artefact
python convert_to_onnx.py        # -> models/bilstm_model.onnx + bilstm_vocab.json

# 6. Rules-layer regression check (Node, no Firebase/ONNX/API key needed)
node evaluation/evaluate_chatbot_router.js
```

`requirements.txt` fija `torch>=2.0.0` sin índice de CUDA; instala primero la wheel que corresponda
a tu propia versión de CUDA si quieres entrenar en GPU.

### Reproducir la medición de la fuga

La auditoría de la sección 4 es la parte que merece la pena reejecutar. Esto es todo:

```python
import json
from collections import defaultdict

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def skeleton(ex):
    """Substitute entity values back into their placeholders."""
    t = ex["text"]
    for k, v in sorted(ex["entities"].items(), key=lambda kv: -len(str(kv[1]))):
        v = str(v)
        if v and v in t:
            t = t.replace(v, "{" + k + "}")
    return (ex["intent"], t)

train, test = load("dataset/train.jsonl"), load("dataset/test.jsonl")

seen = {skeleton(e) for e in train}
hits = sum(1 for e in test if skeleton(e) in seen)
print(f"template leakage: {hits}/{len(test)} = {hits/len(test)*100:.2f}%")
# -> template leakage: 11686/11845 = 98.66%

vocab = {w for e in train for w in e["text"].lower().split()}
toks = [t for e in test for t in e["text"].lower().split()]
oov = [t for t in toks if t not in vocab]
print(f"OOV tokens: {len(oov)}/{len(toks)} = {len(oov)/len(toks)*100:.4f}%")
# -> OOV tokens: 4/103574 = 0.0039%

labels = defaultdict(set)
for e in train + load("dataset/val.jsonl") + test:
    labels[e["text"].lower()].add(e["intent"])
print(f"contradictory labels: {sum(1 for v in labels.values() if len(v) > 1)}")
# -> contradictory labels: 87
```

---

## Estructura del repositorio

```
tfm/                                  (branch: tfm)
  build_dataset.py                    # 969 lines. Template generator, seed 42.
  generate_dataset.py                 # LLM-based alternative. Written, never executed.
  train_{naive_bayes,random_forest,svm,bilstm,bert}.py
  convert_to_onnx.py                  # PyTorch -> ONNX opset 14
  compare_models.py
  results/*.json                      # Per-model metrics, per-intent P/R/F1, confusion matrices
  evaluation/
    chatbot_intent_cases.jsonl        # 26 curated routing cases
    evaluate_chatbot_router.js        # Imports the deployed router module directly

functions/
  chatbot/router.js                   # 234 lines. Pure. Rules + normalisers + thresholds.
  chatbot/bilstm_model.onnx           # 9.7 MiB
  chatbot/bilstm_vocab.json           # 22 KB
  index.js:3440-4400                  # ONNX session, classifier, entity extraction, actions
                                      # (of 4,684 lines total)

lib/
  services/chatbot_service.dart       # 81 lines
  screens/home/dashboard/chatbot_screen.dart   # 448 lines
```

---

## Nota final

El número que defendería no es el 99,57%. Es este: cinco familias de modelos muy distintas quedaron
a 21 errores unas de otras y, en lugar de leer esa uniformidad como una confirmación, medí el propio
benchmark hasta poder explicar de dónde salía la cifra. Salía de un shuffle plano sobre datos
generados con plantillas —un 98,66% de fuga de plantillas— y de una colisión de plantillas que
escribí yo mismo, que explica 52 de los 53 errores restantes del modelo ganador.

La misma pasada sobre el despliegue encontró un 42,5% de desajuste entre el tokenizador de
entrenamiento y el de servicio, que es la razón de que el chatbot siga en una rama.