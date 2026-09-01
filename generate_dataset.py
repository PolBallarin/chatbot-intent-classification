#!/usr/bin/env python3
"""
Script para generar dataset de entrenamiento usando Claude API.
Genera ejemplos de frases con intent + entidades para cada intent definido.

Uso:
    ANTHROPIC_API_KEY=sk-... python generate_dataset.py
    ANTHROPIC_API_KEY=sk-... python generate_dataset.py add_medication
    (para generar solo un intent específico)
"""

import anthropic
import json
import os
import sys
import time
import re
from pathlib import Path

# Cargar .env si existe
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().strip().splitlines():
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()

client = anthropic.Anthropic()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE INTENTS
# ══════════════════════════════════════════════════════════════════════════════

INTENTS = {
    "add_medication": {
        "target": 20000,
        "batch_size": 100,
        "description": "El usuario quiere AÑADIR un nuevo medicamento a una persona",
        "required_entities": ["medication_name", "person"],
        "optional_entities": ["dose", "frequency", "time"],
        "examples": [
            {"text": "Ponle ibuprofeno a mi abuela por las mañanas", "entities": {"medication_name": "ibuprofeno", "person": "mi abuela", "time": "por las mañanas"}},
            {"text": "Añade paracetamol 500mg a Juan cada 8 horas", "entities": {"medication_name": "paracetamol", "person": "Juan", "dose": "500mg", "frequency": "cada 8 horas"}},
            {"text": "Mi madre tiene que tomar omeprazol antes de desayunar", "entities": {"medication_name": "omeprazol", "person": "mi madre", "time": "antes de desayunar"}},
            {"text": "Apunta que la yaya necesita aspirina una vez al día", "entities": {"medication_name": "aspirina", "person": "la yaya", "frequency": "una vez al día"}},
            {"text": "Pon la pastilla azul al abuelo a las 9 y a las 21", "entities": {"medication_name": "la pastilla azul", "person": "el abuelo", "time": "a las 9 y a las 21"}},
        ],
        "entity_examples": {
            "medication_name": ["ibuprofeno", "paracetamol", "omeprazol", "aspirina", "amoxicilina", "lorazepam", "enalapril", "metformina", "simvastatina", "levotiroxina", "la pastilla azul", "la pastilla blanca", "las gotas", "el jarabe", "la crema", "el inhalador", "las vitaminas", "el hierro", "el calcio", "la insulina", "el protector de estómago", "la pastilla de la tensión", "la pastilla del corazón", "el antibiótico"],
            "person": ["mi abuela", "mi abuelo", "la yaya", "el yayo", "mamá", "papá", "mi madre", "mi padre", "la abuela María", "el abuelo Juan", "Carmen", "Antonio", "Dolores", "Francisco", "Pilar", "José", "Mercedes", "Manuel", "Rosa", "Pedro", "mi suegra", "mi suegro", "la tía Rosa"],
            "dose": ["500mg", "1g", "100mg", "250mg", "una pastilla", "dos pastillas", "medio comprimido", "media pastilla", "20mg", "10ml", "5ml", "una cucharada", "dos gotas", "un sobre", "una cápsula"],
            "frequency": ["cada 8 horas", "cada 12 horas", "cada 24 horas", "una vez al día", "dos veces al día", "tres veces al día", "todos los días", "lunes miércoles y viernes", "cada dos días", "una vez a la semana", "los fines de semana"],
            "time": ["por la mañana", "por la tarde", "por la noche", "a las 8", "a las 9", "a las 14", "a las 21", "antes de desayunar", "después de comer", "antes de dormir", "con el desayuno", "en ayunas", "antes de cenar", "después de cenar", "al mediodía"],
        },
    },

    "mark_medication_taken": {
        "target": 15000,
        "batch_size": 100,
        "description": "El usuario quiere MARCAR un medicamento como tomado",
        "required_entities": ["medication_name", "person"],
        "optional_entities": ["time"],
        "examples": [
            {"text": "Mi abuela ya se ha tomado el ibuprofeno", "entities": {"medication_name": "ibuprofeno", "person": "mi abuela"}},
            {"text": "Juan ya tomó la pastilla de la mañana", "entities": {"medication_name": "la pastilla", "person": "Juan", "time": "de la mañana"}},
            {"text": "Marca como tomado el paracetamol de la yaya", "entities": {"medication_name": "paracetamol", "person": "la yaya"}},
            {"text": "El abuelo acaba de tomarse la pastilla de la tensión", "entities": {"medication_name": "la pastilla de la tensión", "person": "el abuelo"}},
            {"text": "Ya le he dado el jarabe a mamá esta noche", "entities": {"medication_name": "el jarabe", "person": "mamá", "time": "esta noche"}},
        ],
        "entity_examples": {
            "medication_name": ["ibuprofeno", "paracetamol", "omeprazol", "la pastilla", "el jarabe", "las gotas", "la pastilla de la tensión", "el antibiótico", "la insulina", "las vitaminas", "el hierro", "la pastilla azul", "el omeprazol", "la aspirina"],
            "person": ["mi abuela", "mi abuelo", "la yaya", "el yayo", "mamá", "papá", "mi madre", "mi padre", "Carmen", "Antonio", "Juan", "la abuela", "el abuelo"],
            "time": ["de la mañana", "de la tarde", "de la noche", "del mediodía", "esta mañana", "esta tarde", "esta noche", "ahora", "hace un rato"],
        },
    },

    "list_medications": {
        "target": 10000,
        "batch_size": 100,
        "description": "El usuario quiere VER o CONSULTAR los medicamentos de una persona",
        "required_entities": ["person"],
        "optional_entities": ["date"],
        "examples": [
            {"text": "¿Qué medicamentos toma mi abuela hoy?", "entities": {"person": "mi abuela", "date": "hoy"}},
            {"text": "Dime las pastillas de Juan", "entities": {"person": "Juan"}},
            {"text": "¿Qué tiene que tomar la yaya mañana?", "entities": {"person": "la yaya", "date": "mañana"}},
            {"text": "Muéstrame la medicación del abuelo", "entities": {"person": "el abuelo"}},
            {"text": "¿Cuáles son los medicamentos de mamá para hoy?", "entities": {"person": "mamá", "date": "hoy"}},
        ],
        "entity_examples": {
            "person": ["mi abuela", "mi abuelo", "la yaya", "el yayo", "mamá", "papá", "Carmen", "Antonio", "Juan", "la abuela", "el abuelo", "mi madre"],
            "date": ["hoy", "mañana", "el lunes", "esta semana", "el jueves", "pasado mañana"],
        },
    },

    "check_medication_status": {
        "target": 10000,
        "batch_size": 100,
        "description": "El usuario quiere COMPROBAR si una persona se ha tomado sus medicamentos",
        "required_entities": ["person"],
        "optional_entities": ["date"],
        "examples": [
            {"text": "¿Se ha tomado todo mi abuela hoy?", "entities": {"person": "mi abuela", "date": "hoy"}},
            {"text": "¿Le falta algo por tomar a Juan?", "entities": {"person": "Juan"}},
            {"text": "¿La yaya ha tomado todas sus pastillas?", "entities": {"person": "la yaya"}},
            {"text": "Comprueba si el abuelo se ha tomado la medicación", "entities": {"person": "el abuelo"}},
            {"text": "¿Qué le queda por tomar a mamá?", "entities": {"person": "mamá"}},
        ],
        "entity_examples": {
            "person": ["mi abuela", "mi abuelo", "la yaya", "el yayo", "mamá", "papá", "Carmen", "Antonio", "Juan", "la abuela", "el abuelo", "mi madre"],
            "date": ["hoy", "esta mañana", "esta tarde", "ayer"],
        },
    },

    "add_reminder": {
        "target": 20000,
        "batch_size": 100,
        "description": "El usuario quiere CREAR un recordatorio o cita en el calendario",
        "required_entities": ["reminder_title", "date", "time", "person"],
        "optional_entities": ["description"],
        "examples": [
            {"text": "Ponle cita al médico a mi abuela el jueves a las 10", "entities": {"reminder_title": "cita al médico", "person": "mi abuela", "date": "el jueves", "time": "a las 10"}},
            {"text": "Añade recordatorio de analítica para Juan el 20 de marzo a las 9", "entities": {"reminder_title": "analítica", "person": "Juan", "date": "el 20 de marzo", "time": "a las 9"}},
            {"text": "Apunta que la yaya tiene revisión el lunes a las 12 en el hospital", "entities": {"reminder_title": "revisión", "person": "la yaya", "date": "el lunes", "time": "a las 12", "description": "en el hospital"}},
            {"text": "Pon recordatorio para mamá: dentista mañana a las 16:30", "entities": {"reminder_title": "dentista", "person": "mamá", "date": "mañana", "time": "a las 16:30"}},
            {"text": "Crea una cita de podólogo para el abuelo el viernes por la mañana", "entities": {"reminder_title": "podólogo", "person": "el abuelo", "date": "el viernes", "time": "por la mañana"}},
        ],
        "entity_examples": {
            "reminder_title": ["cita al médico", "analítica", "revisión", "dentista", "podólogo", "fisioterapia", "vacuna", "ecografía", "radiografía", "consulta", "rehabilitación", "oculista", "cardiólogo", "dermatólogo", "cita con la enfermera", "análisis de sangre", "revisión anual", "control de tensión"],
            "person": ["mi abuela", "mi abuelo", "la yaya", "el yayo", "mamá", "papá", "Carmen", "Antonio", "Juan", "la abuela", "el abuelo", "mi madre"],
            "date": ["mañana", "el lunes", "el martes", "el jueves", "el viernes", "el 20 de marzo", "el 5 de abril", "la semana que viene", "pasado mañana", "el día 15", "este miércoles"],
            "time": ["a las 9", "a las 10", "a las 10:30", "a las 12", "a las 16", "a las 16:30", "a las 17", "por la mañana", "por la tarde", "a mediodía"],
            "description": ["en el hospital", "en el centro de salud", "llevar informes", "en ayunas", "llevar la tarjeta sanitaria", "en la planta 3", "con el Dr. García"],
        },
    },

    "list_reminders": {
        "target": 8000,
        "batch_size": 100,
        "description": "El usuario quiere VER los recordatorios o citas pendientes",
        "required_entities": [],
        "optional_entities": ["person", "date"],
        "examples": [
            {"text": "¿Qué citas tiene mi abuela esta semana?", "entities": {"person": "mi abuela", "date": "esta semana"}},
            {"text": "Dime los recordatorios de mañana", "entities": {"date": "mañana"}},
            {"text": "¿Qué hay pendiente para Juan?", "entities": {"person": "Juan"}},
            {"text": "¿Hay algo programado para hoy?", "entities": {"date": "hoy"}},
            {"text": "Muéstrame las citas de la yaya", "entities": {"person": "la yaya"}},
        ],
        "entity_examples": {
            "person": ["mi abuela", "mi abuelo", "la yaya", "mamá", "papá", "Carmen", "Juan", "el abuelo"],
            "date": ["hoy", "mañana", "esta semana", "el lunes", "este mes", "la semana que viene"],
        },
    },

    "add_contact": {
        "target": 15000,
        "batch_size": 100,
        "description": "El usuario quiere AÑADIR un contacto (médico, cuidador, etc.) a una persona",
        "required_entities": ["contact_name", "phone", "person"],
        "optional_entities": [],
        "examples": [
            {"text": "Añade al Dr. García con teléfono 612345678 a mi abuela", "entities": {"contact_name": "Dr. García", "phone": "612345678", "person": "mi abuela"}},
            {"text": "Guarda el contacto de la enfermera María 698765432 para Juan", "entities": {"contact_name": "enfermera María", "phone": "698765432", "person": "Juan"}},
            {"text": "Pon el número del cardiólogo 934567890 en los contactos de la yaya", "entities": {"contact_name": "cardiólogo", "phone": "934567890", "person": "la yaya"}},
            {"text": "Apunta el teléfono de la farmacia 912345678 para el abuelo", "entities": {"contact_name": "la farmacia", "phone": "912345678", "person": "el abuelo"}},
            {"text": "Añade a la fisioterapeuta Laura 654321987 para mamá", "entities": {"contact_name": "fisioterapeuta Laura", "phone": "654321987", "person": "mamá"}},
        ],
        "entity_examples": {
            "contact_name": ["Dr. García", "Dra. López", "enfermera María", "la farmacia", "el cardiólogo", "el fisioterapeuta", "la cuidadora Ana", "el Dr. Martínez", "el centro de salud", "urgencias", "la ambulancia", "el hospital", "la residencia", "la podóloga", "el dentista", "la óptica"],
            "phone": ["612345678", "698765432", "934567890", "654321987", "912345678", "687654321", "623456789", "911234567", "676543210", "645678901"],
            "person": ["mi abuela", "mi abuelo", "la yaya", "el yayo", "mamá", "papá", "Carmen", "Antonio", "Juan", "la abuela", "el abuelo"],
        },
    },

    "add_shopping_item": {
        "target": 15000,
        "batch_size": 100,
        "description": "El usuario quiere AÑADIR un producto a la lista de la compra",
        "required_entities": ["product"],
        "optional_entities": ["quantity", "store"],
        "examples": [
            {"text": "Añade tomates a la lista de la compra", "entities": {"product": "tomates"}},
            {"text": "Apunta para comprar espárragos en el Mercadona", "entities": {"product": "espárragos", "store": "Mercadona"}},
            {"text": "Pon 2 kilos de naranjas en la lista", "entities": {"product": "naranjas", "quantity": "2 kilos"}},
            {"text": "Necesitamos leche y pan", "entities": {"product": "leche y pan"}},
            {"text": "Compra pañales en la farmacia", "entities": {"product": "pañales", "store": "la farmacia"}},
        ],
        "entity_examples": {
            "product": ["tomates", "leche", "pan", "huevos", "naranjas", "manzanas", "pollo", "arroz", "pasta", "aceite", "agua", "yogures", "queso", "jamón", "papel higiénico", "jabón", "pañales", "espárragos", "patatas", "cebolla", "ajo", "plátanos", "galletas", "café", "azúcar", "sal", "mantequilla", "zumo"],
            "quantity": ["1 kilo", "2 kilos", "medio kilo", "una docena", "un litro", "2 litros", "un paquete", "dos paquetes", "una caja", "3", "6", "una bolsa", "250 gramos"],
            "store": ["Mercadona", "Lidl", "Carrefour", "Aldi", "el Día", "la farmacia", "el mercado", "la frutería", "la carnicería", "la panadería", "el súper", "el Corte Inglés"],
        },
    },

    "mark_shopping_done": {
        "target": 10000,
        "batch_size": 100,
        "description": "El usuario quiere MARCAR un producto como comprado o quitarlo de la lista",
        "required_entities": ["product"],
        "optional_entities": [],
        "examples": [
            {"text": "Ya he comprado los tomates", "entities": {"product": "tomates"}},
            {"text": "Quita la leche de la lista", "entities": {"product": "leche"}},
            {"text": "Ya tenemos el pan", "entities": {"product": "pan"}},
            {"text": "Tacha los huevos", "entities": {"product": "huevos"}},
            {"text": "Marca como comprado el arroz", "entities": {"product": "arroz"}},
        ],
        "entity_examples": {
            "product": ["tomates", "leche", "pan", "huevos", "naranjas", "pollo", "arroz", "pasta", "aceite", "agua", "yogures", "queso", "pañales", "jabón", "café", "galletas", "patatas", "fruta"],
        },
    },

    "list_shopping": {
        "target": 5000,
        "batch_size": 100,
        "description": "El usuario quiere VER la lista de la compra",
        "required_entities": [],
        "optional_entities": [],
        "examples": [
            {"text": "¿Qué hay en la lista de la compra?", "entities": {}},
            {"text": "¿Qué tenemos que comprar?", "entities": {}},
            {"text": "Dime la lista de la compra", "entities": {}},
            {"text": "Enséñame lo que falta por comprar", "entities": {}},
            {"text": "¿Qué necesitamos del súper?", "entities": {}},
        ],
        "entity_examples": {},
    },

    "greeting": {
        "target": 3000,
        "batch_size": 100,
        "description": "El usuario saluda o inicia conversación",
        "required_entities": [],
        "optional_entities": [],
        "examples": [
            {"text": "Hola", "entities": {}},
            {"text": "Buenos días", "entities": {}},
            {"text": "Buenas tardes", "entities": {}},
            {"text": "Qué tal", "entities": {}},
            {"text": "Hey", "entities": {}},
        ],
        "entity_examples": {},
    },

    "help": {
        "target": 3000,
        "batch_size": 100,
        "description": "El usuario pregunta qué puede hacer el chatbot o pide ayuda",
        "required_entities": [],
        "optional_entities": [],
        "examples": [
            {"text": "¿Qué puedes hacer?", "entities": {}},
            {"text": "Ayuda", "entities": {}},
            {"text": "¿Cómo funciona esto?", "entities": {}},
            {"text": "¿Para qué sirves?", "entities": {}},
            {"text": "¿Qué opciones tengo?", "entities": {}},
        ],
        "entity_examples": {},
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# PROMPT PARA GENERAR EJEMPLOS
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(intent_name, intent_config, batch_number, total_batches):
    entity_examples_str = "\n".join(
        f"  - {entity}: {', '.join(values)}"
        for entity, values in intent_config.get("entity_examples", {}).items()
    )

    seed_examples_str = "\n".join(
        json.dumps(ex, ensure_ascii=False) for ex in intent_config["examples"]
    )

    all_entities = intent_config["required_entities"] + intent_config["optional_entities"]
    if all_entities:
        entities_description = f"""Entidades a extraer: {", ".join(all_entities)}
Obligatorias (siempre presentes): {", ".join(intent_config["required_entities"]) or "ninguna"}
Opcionales (a veces presentes, a veces no): {", ".join(intent_config["optional_entities"]) or "ninguna"}"""
    else:
        entities_description = "Este intent NO tiene entidades, solo texto."

    return f"""Genera exactamente {intent_config["batch_size"]} ejemplos ÚNICOS de frases en CASTELLANO para el intent "{intent_name}".

Descripción del intent: {intent_config["description"]}

{entities_description}

Valores de ejemplo para las entidades (usa estos y INVENTA MUCHOS MÁS variados):
{entity_examples_str or "No hay entidades."}

Ejemplos de referencia (NO los repitas, genera NUEVOS):
{seed_examples_str}

REGLAS IMPORTANTES:
1. Cada frase debe ser DIFERENTE y NATURAL, como la diría una persona real en España
2. Varía el registro: formal, informal, coloquial, con tuteo, con usted
3. Varía la estructura: preguntas, órdenes, peticiones, afirmaciones
4. Incluye errores ortográficos ocasionales (10% de las frases): "ponle", "pon le", "k", "xfa", "porfavor", "q"
5. Incluye frases cortas y largas
6. Incluye frases con y sin todas las entidades opcionales
7. Para la entidad "person" usa nombres propios españoles variados, parentescos coloquiales (la yaya, el yayo, tita, tito, la abu, el abu, etc.)
8. No repitas la misma estructura con solo cambiar un nombre - cambia la frase entera
9. Este es el lote {batch_number}/{total_batches}, sé especialmente creativo para no repetir patrones anteriores
10. IMPORTANTE: Genera SOLO frases que correspondan al intent "{intent_name}", no mezcles con otros intents

Responde SOLO con un JSON array, sin texto adicional. Formato:
[
  {{"text": "frase del usuario", "entities": {{"entity_name": "valor extraído de la frase"}}}},
  ...
]

Si el intent no tiene entidades, pon "entities": {{}}
Los valores de las entidades deben ser EXACTAMENTE como aparecen en la frase (mismo texto)."""

# ══════════════════════════════════════════════════════════════════════════════
# GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def generate_batch(intent_name, intent_config, batch_number, total_batches):
    prompt = build_prompt(intent_name, intent_config, batch_number, total_batches)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Extraer JSON del response (puede venir envuelto en ```json ... ```)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_str = json_match.group(1).strip() if json_match else text

    examples = json.loads(json_str)

    # Añadir intent a cada ejemplo
    return [
        {"text": ex["text"], "intent": intent_name, "entities": ex.get("entities", {})}
        for ex in examples
    ]


def count_existing_lines(filepath):
    """Cuenta líneas existentes en un archivo JSONL."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def generate_intent(intent_name, intent_config):
    output_file = f"dataset/{intent_name}.jsonl"
    existing = count_existing_lines(output_file)

    remaining = intent_config["target"] - existing
    if remaining <= 0:
        print(f"✅ {intent_name}: ya tiene {existing}/{intent_config['target']} ejemplos")
        return existing

    total_batches = (remaining + intent_config["batch_size"] - 1) // intent_config["batch_size"]
    print(f"🚀 {intent_name}: generando {remaining} ejemplos en {total_batches} lotes (tiene {existing})")

    generated = existing
    consecutive_errors = 0

    i = 0
    while i < total_batches:
        try:
            batch = generate_batch(intent_name, intent_config, i + 1, total_batches)

            # Append al archivo
            with open(output_file, "a", encoding="utf-8") as f:
                for ex in batch:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")

            generated += len(batch)
            consecutive_errors = 0

            percent = (generated / intent_config["target"]) * 100
            print(f"  📝 Lote {i + 1}/{total_batches}: +{len(batch)} ejemplos ({generated}/{intent_config['target']} = {percent:.1f}%)")

            # Pausa entre lotes
            if i < total_batches - 1:
                time.sleep(0.5)

            i += 1

        except Exception as error:
            consecutive_errors += 1
            print(f"  ❌ Error en lote {i + 1}: {error}")

            if consecutive_errors >= 3:
                print(f"  ⛔ 3 errores consecutivos, parando {intent_name}")
                break

            time.sleep(2)
            # No incrementar i para reintentar

    print(f"✅ {intent_name}: {generated} ejemplos generados")
    return generated


def main():
    # Crear directorio de output
    os.makedirs("dataset", exist_ok=True)

    # Permitir generar un intent específico por argumento
    target_intent = sys.argv[1] if len(sys.argv) > 1 else None
    if target_intent:
        if target_intent not in INTENTS:
            print(f"❌ Intent \"{target_intent}\" no existe. Disponibles: {', '.join(INTENTS.keys())}")
            sys.exit(1)
        intents_to_generate = {target_intent: INTENTS[target_intent]}
    else:
        intents_to_generate = INTENTS

    print("═══════════════════════════════════════════")
    print("  GENERADOR DE DATASET - PLAMILY CHATBOT")
    print("═══════════════════════════════════════════\n")

    total_target = sum(i["target"] for i in intents_to_generate.values())
    print(f"📊 Intents: {len(intents_to_generate)}")
    print(f"🎯 Ejemplos objetivo: {total_target:,}\n")

    total_generated = 0

    for name, config in intents_to_generate.items():
        count = generate_intent(name, config)
        total_generated += count
        print()

    print("═══════════════════════════════════════════")
    print(f"✅ TOTAL: {total_generated:,} ejemplos generados")
    print("═══════════════════════════════════════════")

    # Resumen
    print("\n📋 Resumen por intent:")
    for name in intents_to_generate:
        filepath = f"dataset/{name}.jsonl"
        count = count_existing_lines(filepath)
        target = INTENTS[name]["target"]
        status = "✅" if count >= target else "⏳"
        print(f"  {status} {name}: {count}/{target}")


if __name__ == "__main__":
    main()
