#!/usr/bin/env python3
"""
Generador de dataset para el chatbot de Plamily.
Genera ejemplos combinando templates con entidades aleatorias.
Sin llamadas a API - todo local.

Uso: python3 build_dataset.py
"""

import json
import random
import os

random.seed(42)  # Reproducible

# ══════════════════════════════════════════════════════════════════════════════
# VALORES DE ENTIDADES
# ══════════════════════════════════════════════════════════════════════════════

PERSONS = [
    # Parentescos formales
    "mi abuela", "mi abuelo", "mi madre", "mi padre", "mi suegra", "mi suegro",
    "mi tía", "mi tío", "mi hermana", "mi hermano",
    # Parentescos coloquiales
    "la yaya", "el yayo", "la abu", "el abu", "mamá", "papá",
    "la tita", "el tito", "la nona", "el nono", "la abuelita", "el abuelito",
    # Nombres propios
    "Carmen", "Antonio", "Juan", "María", "Dolores", "Francisco", "Pilar",
    "José", "Mercedes", "Manuel", "Rosa", "Pedro", "Concha", "Paco",
    "Lola", "Manolo", "Pepe", "Luisa", "Fernando", "Isabel", "Ramón",
    "Teresa", "Enrique", "Amparo", "Miguel", "Esperanza", "Rafael",
    "Josefa", "Ángel", "Rosario", "Joaquín", "Margarita", "Andrés",
    "Encarna", "Tomás", "Gloria", "Emilio", "Consuelo", "Vicente",
    # Parentesco + nombre
    "la abuela María", "el abuelo Juan", "la abuela Carmen", "el abuelo Paco",
    "la abuela Lola", "el abuelo Manuel", "la tía Rosa", "el tío Pepe",
    "la abuela Pilar", "el abuelo Antonio", "la abuela Concha", "el abuelo Ramón",
]

MEDICATIONS = [
    # Nombres genéricos
    "ibuprofeno", "paracetamol", "omeprazol", "aspirina", "amoxicilina",
    "lorazepam", "enalapril", "metformina", "simvastatina", "levotiroxina",
    "diclofenaco", "nolotil", "atorvastatina", "bisoprolol", "furosemida",
    "ramipril", "amlodipino", "losartán", "clopidogrel", "prednisona",
    "gabapentina", "tramadol", "pantoprazol", "lansoprazol", "alprazolam",
    "diazepam", "captopril", "hidrolorotiazida", "warfarina", "digoxina",
    "salbutamol", "insulina", "metoclopramida", "domperidona", "ranitidina",
    # Nombres coloquiales
    "la pastilla azul", "la pastilla blanca", "la pastilla rosa",
    "la pastilla pequeña", "la pastilla grande", "la pastilla redonda",
    "las gotas", "el jarabe", "la crema", "el inhalador", "las vitaminas",
    "el hierro", "el calcio", "el protector de estómago", "la pastilla de la tensión",
    "la pastilla del corazón", "el antibiótico", "la pastilla para dormir",
    "el antiinflamatorio", "el analgésico", "la pastilla del azúcar",
    "la pastilla del colesterol", "la pastilla de la tiroides", "el colirio",
    "las cápsulas", "los sobres", "el spray", "la pomada", "los comprimidos",
    "el parche", "las ampollas", "el suplemento",
]

DOSES = [
    "500mg", "1g", "100mg", "250mg", "200mg", "50mg", "75mg", "150mg",
    "10mg", "20mg", "25mg", "40mg", "80mg", "5mg", "2.5mg",
    "una pastilla", "dos pastillas", "media pastilla", "medio comprimido",
    "una cápsula", "dos cápsulas", "un sobre", "medio sobre",
    "5ml", "10ml", "15ml", "una cucharada", "media cucharada",
    "dos gotas", "tres gotas", "cinco gotas", "una ampolla",
    "un comprimido", "dos comprimidos", "un parche",
]

FREQUENCIES = [
    "cada 8 horas", "cada 12 horas", "cada 24 horas", "cada 6 horas",
    "una vez al día", "dos veces al día", "tres veces al día",
    "todos los días", "cada día", "a diario",
    "lunes miércoles y viernes", "lunes y jueves", "martes y viernes",
    "cada dos días", "día sí día no", "en días alternos",
    "una vez a la semana", "dos veces a la semana",
    "los fines de semana", "entre semana", "de lunes a viernes",
    "cada 4 horas", "solo cuando le duela", "cuando lo necesite",
]

TIMES = [
    "por la mañana", "por la tarde", "por la noche",
    "a las 7", "a las 8", "a las 8:30", "a las 9", "a las 9:30",
    "a las 10", "a las 11", "a las 12", "a las 13", "a las 14",
    "a las 15", "a las 16", "a las 17", "a las 18", "a las 19",
    "a las 20", "a las 21", "a las 22", "a las 23",
    "antes de desayunar", "después de desayunar", "con el desayuno",
    "antes de comer", "después de comer", "con la comida",
    "antes de cenar", "después de cenar", "con la cena",
    "antes de dormir", "al acostarse", "al levantarse",
    "en ayunas", "al mediodía", "a media mañana", "a media tarde",
    "de mañana", "de tarde", "de noche",
]

TIMES_TAKEN = [
    "de la mañana", "de la tarde", "de la noche", "del mediodía",
    "esta mañana", "esta tarde", "esta noche", "ahora",
    "hace un rato", "hace un momento", "hace poco",
    "antes de comer", "después de comer", "al levantarse",
]

DATES = [
    "hoy", "mañana", "pasado mañana", "ayer",
    "el lunes", "el martes", "el miércoles", "el jueves", "el viernes",
    "el sábado", "el domingo",
    "este lunes", "este martes", "este miércoles", "este jueves", "este viernes",
    "la semana que viene", "la próxima semana",
    "el día 1", "el día 5", "el día 10", "el día 15", "el día 20", "el día 25",
    "el 1 de abril", "el 5 de abril", "el 10 de abril", "el 15 de marzo",
    "el 20 de marzo", "el 25 de marzo", "el 3 de mayo", "el 12 de junio",
    "esta semana", "este mes",
]

REMINDER_TITLES = [
    "cita al médico", "cita médica", "cita con el médico", "consulta médica",
    "analítica", "análisis de sangre", "análisis",
    "revisión", "revisión anual", "revisión médica", "chequeo",
    "dentista", "cita con el dentista", "revisión dental",
    "podólogo", "cita con el podólogo",
    "fisioterapia", "sesión de fisioterapia", "fisio", "rehabilitación",
    "vacuna", "vacuna de la gripe", "vacunación",
    "ecografía", "radiografía", "resonancia", "TAC", "escáner",
    "oculista", "revisión de la vista", "oftalmólogo",
    "cardiólogo", "cita con el cardiólogo",
    "dermatólogo", "cita con el dermatólogo",
    "cita con la enfermera", "enfermera", "control de enfermería",
    "control de tensión", "control de azúcar", "control de peso",
    "traumatólogo", "urólogo", "neurólogo", "endocrino",
    "logopeda", "psicólogo", "terapia",
]

DESCRIPTIONS = [
    "en el hospital", "en el centro de salud", "en la clínica",
    "llevar informes", "llevar los análisis", "llevar la documentación",
    "en ayunas", "sin desayunar", "no puede comer antes",
    "llevar la tarjeta sanitaria", "llevar el DNI",
    "en la planta 3", "en la planta 2", "en consulta 5", "en consulta 12",
    "con el Dr. García", "con la Dra. López", "con el Dr. Martínez",
    "pedir cita previa antes", "confirmar el día antes",
    "ir acompañado", "necesita silla de ruedas",
    "es urgente", "es revisión rutinaria",
]

CONTACT_NAMES = [
    "Dr. García", "Dra. López", "Dr. Martínez", "Dra. Sánchez",
    "Dr. Fernández", "Dra. Rodríguez", "Dr. Pérez", "Dra. González",
    "Dr. Ruiz", "Dra. Hernández", "Dr. Jiménez", "Dra. Moreno",
    "enfermera María", "enfermero Carlos", "enfermera Ana", "enfermera Laura",
    "la farmacia", "la farmacia del barrio", "la farmacia de guardia",
    "el cardiólogo", "el fisioterapeuta", "el dentista", "el podólogo",
    "la cuidadora Ana", "la cuidadora María", "el cuidador Pedro",
    "el centro de salud", "urgencias", "el hospital", "la ambulancia",
    "la residencia", "la óptica", "la ortopedia",
    "el 112", "el teléfono de emergencias",
    "la trabajadora social", "el asistente social",
    "la podóloga", "la dermatóloga", "el traumatólogo",
    "el taxi adaptado", "el transporte sanitario",
]

PHONES = [
    "612345678", "698765432", "934567890", "654321987", "912345678",
    "687654321", "623456789", "911234567", "676543210", "645678901",
    "633221144", "699887766", "677554433", "688443322", "622115577",
    "611223344", "655443322", "644332211", "666778899", "677889900",
    "933445566", "912233445", "915566778", "934455667", "916677889",
]

PRODUCTS = [
    "tomates", "leche", "pan", "huevos", "naranjas", "manzanas", "plátanos",
    "pollo", "carne picada", "filetes de ternera", "pechugas de pollo",
    "arroz", "pasta", "macarrones", "espaguetis", "fideos",
    "aceite", "aceite de oliva", "agua", "agua mineral",
    "yogures", "queso", "mantequilla", "margarina", "nata",
    "jamón", "jamón york", "jamón serrano", "pavo", "chorizo",
    "papel higiénico", "jabón", "gel de ducha", "champú", "detergente",
    "pañales", "toallitas", "gasas", "esparadrapo",
    "espárragos", "patatas", "cebolla", "ajo", "pimiento", "zanahoria",
    "lechuga", "tomate", "pepino", "calabacín", "berenjena", "judías verdes",
    "galletas", "café", "azúcar", "sal", "harina", "levadura",
    "zumo", "zumo de naranja", "cola", "cerveza", "vino",
    "atún", "sardinas", "merluza", "salmón", "gambas",
    "fruta", "verdura", "carne", "pescado", "embutido",
    "cereales", "tostadas", "mermelada", "miel",
    "chocolate", "helado", "flan", "natillas",
    "lavavajillas", "suavizante", "lejía", "fregona",
    "pilas", "bombillas", "papel de aluminio", "film transparente",
    "servilletas", "bolsas de basura",
]

QUANTITIES = [
    "1 kilo", "2 kilos", "medio kilo", "3 kilos", "un cuarto de kilo",
    "una docena", "media docena", "6", "3", "4", "2", "1", "5", "10",
    "un litro", "2 litros", "medio litro", "un litro y medio",
    "un paquete", "dos paquetes", "tres paquetes",
    "una caja", "dos cajas", "una lata", "dos latas", "tres latas",
    "una bolsa", "dos bolsas",
    "250 gramos", "500 gramos", "100 gramos", "200 gramos",
    "una barra", "dos barras", "una botella", "dos botellas",
    "un bote", "un tarro", "un brick",
]

STORES = [
    "Mercadona", "Lidl", "Carrefour", "Aldi", "el Día", "Eroski",
    "BonÀrea", "Consum", "Alcampo", "Hipercor", "el Corte Inglés",
    "la farmacia", "el mercado", "la frutería", "la carnicería",
    "la panadería", "la pescadería", "el súper", "el hiper",
    "la tienda de abajo", "la tienda del barrio", "la tienda de la esquina",
    "el estanco", "la droguería", "el herbolario",
]


# ══════════════════════════════════════════════════════════════════════════════
# VARIACIONES LINGÜÍSTICAS
# ══════════════════════════════════════════════════════════════════════════════

def maybe_typo(text, prob=0.08):
    """Introduce errores ortográficos ocasionales."""
    if random.random() > prob:
        return text
    replacements = [
        ("que ", "q "), ("que,", "q,"), ("por favor", "porfavor"),
        ("por favor", "porfa"), ("por favor", "xfa"),
        ("también", "tambien"), ("está", "esta"),
        ("día", "dia"), ("médico", "medico"),
        ("teléfono", "telefono"), ("número", "numero"),
        ("añade", "añade"), ("dime", "dme"),
        ("mañana", "mñana"), ("después", "despues"),
        ("cuándo", "cuando"), ("cuánto", "cuanto"),
        ("cómo", "como"), ("qué", "que"),
    ]
    r = random.choice(replacements)
    if r[0] in text.lower():
        return text.replace(r[0], r[1], 1)
    return text


def random_person():
    return random.choice(PERSONS)

def random_med():
    return random.choice(MEDICATIONS)

def random_dose():
    return random.choice(DOSES)

def random_freq():
    return random.choice(FREQUENCIES)

def random_time():
    return random.choice(TIMES)

def random_time_taken():
    return random.choice(TIMES_TAKEN)

def random_date():
    return random.choice(DATES)

def random_reminder():
    return random.choice(REMINDER_TITLES)

def random_desc():
    return random.choice(DESCRIPTIONS)

def random_contact():
    return random.choice(CONTACT_NAMES)

def random_phone():
    return random.choice(PHONES)

def random_product():
    return random.choice(PRODUCTS)

def random_qty():
    return random.choice(QUANTITIES)

def random_store():
    return random.choice(STORES)


# ══════════════════════════════════════════════════════════════════════════════
# GENERADORES POR INTENT
# ══════════════════════════════════════════════════════════════════════════════

def gen_add_medication():
    p = random_person()
    m = random_med()
    d = random_dose()
    f = random_freq()
    t = random_time()

    templates_all = [
        (f"Ponle {m} {d} a {p} {f} {t}", {"medication_name": m, "person": p, "dose": d, "frequency": f, "time": t}),
        (f"Añade {m} de {d} a {p} {t} {f}", {"medication_name": m, "person": p, "dose": d, "frequency": f, "time": t}),
        (f"Que {p} tome {m} {d} {f} {t}", {"medication_name": m, "person": p, "dose": d, "frequency": f, "time": t}),
        (f"Apunta {m} {d} para {p} {t} {f}", {"medication_name": m, "person": p, "dose": d, "frequency": f, "time": t}),
        (f"Necesito que le pongas {m} a {p} {d} {f} {t}", {"medication_name": m, "person": p, "dose": d, "frequency": f, "time": t}),
        (f"Registra {m} {d} para {p} {f} {t}", {"medication_name": m, "person": p, "dose": d, "frequency": f, "time": t}),
    ]

    templates_no_dose = [
        (f"Ponle {m} a {p} {f} {t}", {"medication_name": m, "person": p, "frequency": f, "time": t}),
        (f"Añade {m} a {p} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"{p} tiene que tomar {m} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Apunta que {p} necesita {m} {f}", {"medication_name": m, "person": p, "frequency": f}),
        (f"Pon {m} al tratamiento de {p} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Hay que darle {m} a {p} {f}", {"medication_name": m, "person": p, "frequency": f}),
        (f"Añádele {m} a {p} {f} {t}", {"medication_name": m, "person": p, "frequency": f, "time": t}),
        (f"Mete {m} en la medicación de {p} {t}", {"medication_name": m, "person": p, "time": t}),
    ]

    templates_minimal = [
        (f"Ponle {m} a {p}", {"medication_name": m, "person": p}),
        (f"Añade {m} a {p}", {"medication_name": m, "person": p}),
        (f"Pon {m} para {p}", {"medication_name": m, "person": p}),
        (f"{p} necesita {m}", {"medication_name": m, "person": p}),
        (f"Apunta {m} para {p}", {"medication_name": m, "person": p}),
        (f"Dale {m} a {p}", {"medication_name": m, "person": p}),
        (f"Nuevo medicamento para {p}: {m}", {"medication_name": m, "person": p}),
        (f"Añadir {m} a {p}", {"medication_name": m, "person": p}),
    ]

    templates_conversational = [
        (f"Oye, ponle {m} a {p} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Perdona, ¿puedes añadir {m} a {p}?", {"medication_name": m, "person": p}),
        (f"¿Me puedes poner {m} a {p} {f}?", {"medication_name": m, "person": p, "frequency": f}),
        (f"Necesitaría añadir {m} para {p} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Hola, quiero poner {m} a {p} {f}", {"medication_name": m, "person": p, "frequency": f}),
        (f"Por favor, añade {m} {d} a {p}", {"medication_name": m, "person": p, "dose": d}),
        (f"Le han recetado {m} a {p} {f} {t}", {"medication_name": m, "person": p, "frequency": f, "time": t}),
        (f"El médico le ha mandado {m} a {p} {d}", {"medication_name": m, "person": p, "dose": d}),
        (f"A {p} le han puesto {m} {f}", {"medication_name": m, "person": p, "frequency": f}),
        (f"Tengo que meter {m} a {p} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Hay que añadir {m} a la ficha de {p}", {"medication_name": m, "person": p}),
        (f"Pon en la lista de {p} {m} {d} {t}", {"medication_name": m, "person": p, "dose": d, "time": t}),
        (f"¿Puedes registrar {m} para {p}? Es {f}", {"medication_name": m, "person": p, "frequency": f}),
        (f"A ver, ponle a {p} {m} {d}", {"medication_name": m, "person": p, "dose": d}),
        (f"El doctor ha dicho que {p} tome {m} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Nueva medicación para {p}: {m} {d} {f}", {"medication_name": m, "person": p, "dose": d, "frequency": f}),
    ]

    all_templates = templates_all + templates_no_dose + templates_minimal + templates_conversational
    text, entities = random.choice(all_templates)
    return maybe_typo(text), entities


def gen_mark_medication_taken():
    p = random_person()
    m = random_med()
    t = random_time_taken()

    templates_with_time = [
        (f"{p} ya se ha tomado {m} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Marca como tomado {m} de {p} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Ya le he dado {m} a {p} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"{p} acaba de tomar {m} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Apunta que {p} ya tomó {m} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"{p} se ha tomado {m} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Ya está tomado {m} de {p} {t}", {"medication_name": m, "person": p, "time": t}),
    ]

    templates_no_time = [
        (f"{p} ya se ha tomado {m}", {"medication_name": m, "person": p}),
        (f"Marca como tomado {m} de {p}", {"medication_name": m, "person": p}),
        (f"Ya le he dado {m} a {p}", {"medication_name": m, "person": p}),
        (f"{p} ya tomó {m}", {"medication_name": m, "person": p}),
        (f"Hecho, {p} se ha tomado {m}", {"medication_name": m, "person": p}),
        (f"{m} de {p} ya está tomado", {"medication_name": m, "person": p}),
        (f"{p} acaba de tomarse {m}", {"medication_name": m, "person": p}),
        (f"Ya le di {m} a {p}", {"medication_name": m, "person": p}),
        (f"Pon que {p} ya se tomó {m}", {"medication_name": m, "person": p}),
        (f"Marca {m} de {p}", {"medication_name": m, "person": p}),
        (f"Confirma que {p} tomó {m}", {"medication_name": m, "person": p}),
    ]

    templates_conversational = [
        (f"Oye, {p} ya se ha tomado {m}", {"medication_name": m, "person": p}),
        (f"Te digo que {p} ya tomó {m}", {"medication_name": m, "person": p}),
        (f"¿Puedes marcar {m} de {p} como tomado?", {"medication_name": m, "person": p}),
        (f"Aviso de que {p} se ha tomado {m} {t}", {"medication_name": m, "person": p, "time": t}),
        (f"Hecho lo de {m} de {p}", {"medication_name": m, "person": p}),
        (f"Listo, {p} ha tomado {m}", {"medication_name": m, "person": p}),
        (f"{m} tomado por {p}", {"medication_name": m, "person": p}),
        (f"Ya, {p} se lo ha tomado, {m}", {"medication_name": m, "person": p}),
        (f"Le acabo de dar {m} a {p}", {"medication_name": m, "person": p}),
        (f"Confirmo que {p} ha tomado {m} {t}", {"medication_name": m, "person": p, "time": t}),
    ]

    all_templates = templates_with_time + templates_no_time + templates_conversational
    text, entities = random.choice(all_templates)
    return maybe_typo(text), entities


def gen_list_medications():
    p = random_person()
    d = random_date()
    m = random_med()

    templates_with_date = [
        (f"¿Qué medicamentos toma {p} {d}?", {"person": p, "date": d}),
        (f"Dime la medicación de {p} para {d}", {"person": p, "date": d}),
        (f"¿Qué tiene que tomar {p} {d}?", {"person": p, "date": d}),
        (f"Medicamentos de {p} para {d}", {"person": p, "date": d}),
        (f"¿Cuáles son las pastillas de {p} {d}?", {"person": p, "date": d}),
        (f"Enséñame lo que toma {p} {d}", {"person": p, "date": d}),
        (f"Lista de medicamentos de {p} {d}", {"person": p, "date": d}),
        (f"¿Qué le toca a {p} {d}?", {"person": p, "date": d}),
        (f"Pastillas de {p} para {d}", {"person": p, "date": d}),
        (f"¿Qué se tiene que tomar {p} {d}?", {"person": p, "date": d}),
        (f"Dime qué toma {p} {d}", {"person": p, "date": d}),
        (f"¿Qué medicinas tiene {p} {d}?", {"person": p, "date": d}),
        (f"Consulta la medicación de {p} para {d}", {"person": p, "date": d}),
        (f"¿Cuántas pastillas tiene {p} {d}?", {"person": p, "date": d}),
        (f"Tratamiento de {p} para {d}", {"person": p, "date": d}),
    ]

    templates_no_date = [
        (f"¿Qué medicamentos toma {p}?", {"person": p}),
        (f"Dime las pastillas de {p}", {"person": p}),
        (f"Muéstrame la medicación de {p}", {"person": p}),
        (f"¿Qué toma {p}?", {"person": p}),
        (f"Medicación de {p}", {"person": p}),
        (f"¿Cuáles son los medicamentos de {p}?", {"person": p}),
        (f"¿Qué pastillas tiene {p}?", {"person": p}),
        (f"La medicación de {p}", {"person": p}),
        (f"Dame la lista de medicamentos de {p}", {"person": p}),
        (f"¿Me dices qué toma {p}?", {"person": p}),
        (f"¿Qué le toca tomar a {p}?", {"person": p}),
        (f"Enséñame las medicinas de {p}", {"person": p}),
        (f"Ver medicamentos de {p}", {"person": p}),
        (f"¿Toma {p} {m}?", {"person": p}),
        (f"¿{p} toma algo?", {"person": p}),
        (f"Dime el tratamiento de {p}", {"person": p}),
        (f"¿Qué medicinas toma {p}?", {"person": p}),
        (f"Oye, ¿qué toma {p}?", {"person": p}),
        (f"¿Puedes decirme la medicación de {p}?", {"person": p}),
        (f"A ver, ¿qué toma {p}?", {"person": p}),
        (f"Necesito saber qué medicamentos tiene {p}", {"person": p}),
        (f"¿Cuántos medicamentos toma {p}?", {"person": p}),
        (f"Consultar medicación de {p}", {"person": p}),
        (f"Listado de pastillas de {p}", {"person": p}),
        (f"¿Me enseñas lo que toma {p}?", {"person": p}),
    ]

    all_templates = templates_with_date + templates_no_date
    text, entities = random.choice(all_templates)
    return maybe_typo(text), entities


def gen_check_medication_status():
    p = random_person()
    d = random_date()

    templates_with_date = [
        (f"¿Se ha tomado todo {p} {d}?", {"person": p, "date": d}),
        (f"¿{p} ha tomado todas sus pastillas {d}?", {"person": p, "date": d}),
        (f"¿Le falta algo por tomar a {p} {d}?", {"person": p, "date": d}),
        (f"Comprueba si {p} se ha tomado la medicación {d}", {"person": p, "date": d}),
        (f"¿Cómo va {p} con la medicación {d}?", {"person": p, "date": d}),
    ]

    templates_no_date = [
        (f"¿Se ha tomado todo {p}?", {"person": p}),
        (f"¿{p} ha tomado todas sus pastillas?", {"person": p}),
        (f"¿Le falta algo por tomar a {p}?", {"person": p}),
        (f"Comprueba si {p} se ha tomado todo", {"person": p}),
        (f"¿Qué le queda por tomar a {p}?", {"person": p}),
        (f"¿{p} se ha tomado la medicación?", {"person": p}),
        (f"¿Ha tomado todo {p}?", {"person": p}),
        (f"¿Cómo va {p} con sus pastillas?", {"person": p}),
        (f"Estado de la medicación de {p}", {"person": p}),
        (f"¿Tiene {p} todo tomado?", {"person": p}),
        (f"Mira si {p} se ha tomado todo", {"person": p}),
        (f"¿Le queda algo a {p}?", {"person": p}),
        (f"Dime si {p} ha tomado la medicación", {"person": p}),
        (f"¿{p} lleva todo al día?", {"person": p}),
        (f"Revisa la medicación de {p}", {"person": p}),
        (f"¿{p} se lo ha tomado todo?", {"person": p}),
        (f"Oye, ¿{p} se ha tomado las pastillas?", {"person": p}),
        (f"¿Cómo va la medicación de {p}?", {"person": p}),
        (f"¿{p} ha cumplido con la medicación?", {"person": p}),
        (f"¿Falta algo de {p}?", {"person": p}),
        (f"Verifica si {p} tomó todo", {"person": p}),
        (f"Control de {p}", {"person": p}),
        (f"¿{p} está al día con las pastillas?", {"person": p}),
        (f"¿Le queda medicación a {p}?", {"person": p}),
        (f"¿Ha tomado {p} lo que le tocaba?", {"person": p}),
        (f"¿Cumplió {p} con la medicación?", {"person": p}),
        (f"Mira la medicación de {p}", {"person": p}),
        (f"A ver si {p} se ha tomado todo", {"person": p}),
        (f"¿Le han dado todo a {p}?", {"person": p}),
        (f"¿Qué tal va {p} con los medicamentos?", {"person": p}),
    ]

    all_templates = templates_with_date + templates_no_date
    text, entities = random.choice(all_templates)
    return maybe_typo(text), entities


def gen_add_reminder():
    p = random_person()
    r = random_reminder()
    d = random_date()
    t = random_time()
    desc = random_desc()

    templates_all = [
        (f"Ponle {r} a {p} {d} {t} {desc}", {"reminder_title": r, "person": p, "date": d, "time": t, "description": desc}),
        (f"Añade recordatorio de {r} para {p} {d} {t} {desc}", {"reminder_title": r, "person": p, "date": d, "time": t, "description": desc}),
        (f"Apunta que {p} tiene {r} {d} {t} {desc}", {"reminder_title": r, "person": p, "date": d, "time": t, "description": desc}),
    ]

    templates_no_desc = [
        (f"Ponle {r} a {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Añade {r} para {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Apunta {r} de {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Pon recordatorio: {r} para {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Crea una cita de {r} para {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"{p} tiene {r} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Nuevo recordatorio para {p}: {r} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Agenda {r} de {p} para {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Hay que apuntar {r} de {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
    ]

    templates_partial = [
        (f"Ponle {r} a {p} {d}", {"reminder_title": r, "person": p, "date": d}),
        (f"Añade {r} para {p} {t}", {"reminder_title": r, "person": p, "time": t}),
        (f"{p} tiene {r} {d}", {"reminder_title": r, "person": p, "date": d}),
        (f"Pon {r} a {p}", {"reminder_title": r, "person": p}),
        (f"Apunta {r} de {p}", {"reminder_title": r, "person": p}),
    ]

    templates_conversational = [
        (f"Oye, pon {r} a {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"¿Puedes apuntar {r} de {p} para {d}?", {"reminder_title": r, "person": p, "date": d}),
        (f"Me acaban de decir que {p} tiene {r} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Recordatorio: {r} de {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"No se me olvide: {r} de {p} {d}", {"reminder_title": r, "person": p, "date": d}),
        (f"Acuérdame que {p} tiene {r} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
        (f"Avísame de que {p} tiene {r} {d}", {"reminder_title": r, "person": p, "date": d}),
        (f"Por favor, apunta {r} para {p} {d} {t}", {"reminder_title": r, "person": p, "date": d, "time": t}),
    ]

    all_templates = templates_all + templates_no_desc + templates_partial + templates_conversational
    text, entities = random.choice(all_templates)
    return maybe_typo(text), entities


def gen_list_reminders():
    p = random_person()
    d = random_date()
    r = random_reminder()

    templates = [
        (f"¿Qué citas tiene {p} {d}?", {"person": p, "date": d}),
        (f"Dime los recordatorios de {p}", {"person": p}),
        (f"¿Qué hay pendiente para {p}?", {"person": p}),
        (f"¿Hay algo programado {d}?", {"date": d}),
        (f"Muéstrame las citas de {p}", {"person": p}),
        (f"Recordatorios de {p} {d}", {"person": p, "date": d}),
        (f"¿Qué tiene {p} {d}?", {"person": p, "date": d}),
        (f"¿Tiene {p} algo {d}?", {"person": p, "date": d}),
        (f"Ver citas de {p}", {"person": p}),
        (f"¿Qué citas hay {d}?", {"date": d}),
        (f"Citas {d}", {"date": d}),
        (f"Recordatorios {d}", {"date": d}),
        (f"¿Alguna cita {d}?", {"date": d}),
        (f"¿Qué hay apuntado para {p}?", {"person": p}),
        (f"Dime qué tiene {p} pendiente", {"person": p}),
        (f"¿Hay recordatorios para {d}?", {"date": d}),
        (f"Lista de citas de {p}", {"person": p}),
        (f"¿Qué recordatorios hay?", {}),
        (f"¿Hay citas pendientes?", {}),
        (f"Muéstrame los recordatorios", {}),
        (f"¿Tenemos algo pendiente?", {}),
        (f"Ver recordatorios", {}),
        (f"Próximas citas", {}),
        (f"¿Tiene {p} {r} pronto?", {"person": p}),
        (f"¿Cuándo tiene {p} {r}?", {"person": p}),
        (f"¿Hay {r} {d}?", {"date": d}),
        (f"Oye, ¿qué citas tiene {p}?", {"person": p}),
        (f"¿Qué le toca a {p} {d}?", {"person": p, "date": d}),
        (f"Agenda de {p}", {"person": p}),
        (f"Agenda de {p} {d}", {"person": p, "date": d}),
        (f"¿Tiene algo {p} {d}?", {"person": p, "date": d}),
        (f"Consulta las citas de {p}", {"person": p}),
        (f"Dime las citas de {p} para {d}", {"person": p, "date": d}),
        (f"¿Hay algo apuntado {d}?", {"date": d}),
        (f"¿Qué hay programado {d}?", {"date": d}),
        (f"Citas de {p} {d}", {"person": p, "date": d}),
        (f"¿Tiene citas {p}?", {"person": p}),
        (f"Próximas citas de {p}", {"person": p}),
        (f"¿{p} tiene algo pendiente?", {"person": p}),
        (f"¿Cuáles son las citas de {p}?", {"person": p}),
        (f"¿Qué recordatorios tiene {p} {d}?", {"person": p, "date": d}),
        (f"Mira si {p} tiene algo {d}", {"person": p, "date": d}),
        (f"¿Algún recordatorio para {p}?", {"person": p}),
        (f"A ver las citas de {p}", {"person": p}),
        (f"¿Qué tenemos {d}?", {"date": d}),
        (f"¿Hay algo {d}?", {"date": d}),
    ]

    text, entities = random.choice(templates)
    return maybe_typo(text), entities


def gen_add_contact():
    p = random_person()
    c = random_contact()
    ph = random_phone()

    templates = [
        (f"Añade a {c} con teléfono {ph} a {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Guarda el contacto de {c} {ph} para {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Pon el número de {c} {ph} en los contactos de {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Apunta el teléfono de {c} {ph} para {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Añade contacto: {c}, {ph}, para {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Nuevo contacto para {p}: {c} {ph}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Guarda {c} {ph} en la agenda de {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Mete a {c} con número {ph} en los contactos de {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"El teléfono de {c} es {ph}, ponlo en {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Pon a {c} {ph} para {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Contacto nuevo de {p}: {c} {ph}", {"contact_name": c, "phone": ph, "person": p}),
        (f"¿Puedes guardar el {ph} de {c} para {p}?", {"contact_name": c, "phone": ph, "person": p}),
        (f"Apunta: {c} {ph} contacto de {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Necesito añadir a {c} al teléfono de {p}, su número es {ph}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Oye, guarda el {ph} de {c} para {p}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Añádeme a {c} en {p}, teléfono {ph}", {"contact_name": c, "phone": ph, "person": p}),
        (f"Registra el contacto de {c} para {p}: {ph}", {"contact_name": c, "phone": ph, "person": p}),
        (f"El número de {c} de {p} es {ph}", {"contact_name": c, "phone": ph, "person": p}),
    ]

    text, entities = random.choice(templates)
    return maybe_typo(text), entities


def gen_add_shopping_item():
    pr = random_product()
    q = random_qty()
    s = random_store()

    templates_all = [
        (f"Añade {q} de {pr} a la lista, de {s}", {"product": pr, "quantity": q, "store": s}),
        (f"Compra {q} de {pr} en {s}", {"product": pr, "quantity": q, "store": s}),
        (f"Pon {q} de {pr} en la lista, hay que ir a {s}", {"product": pr, "quantity": q, "store": s}),
    ]

    templates_qty = [
        (f"Añade {q} de {pr} a la lista", {"product": pr, "quantity": q}),
        (f"Pon {q} de {pr} en la lista de la compra", {"product": pr, "quantity": q}),
        (f"Compra {q} de {pr}", {"product": pr, "quantity": q}),
        (f"Necesitamos {q} de {pr}", {"product": pr, "quantity": q}),
        (f"Apunta {q} de {pr}", {"product": pr, "quantity": q}),
        (f"Falta {q} de {pr}", {"product": pr, "quantity": q}),
    ]

    templates_store = [
        (f"Compra {pr} en {s}", {"product": pr, "store": s}),
        (f"Apunta {pr} de {s}", {"product": pr, "store": s}),
        (f"Hay que comprar {pr} en {s}", {"product": pr, "store": s}),
        (f"Pon {pr} en la lista, de {s}", {"product": pr, "store": s}),
    ]

    templates_minimal = [
        (f"Añade {pr} a la lista de la compra", {"product": pr}),
        (f"Pon {pr} en la lista", {"product": pr}),
        (f"Compra {pr}", {"product": pr}),
        (f"Necesitamos {pr}", {"product": pr}),
        (f"Falta {pr}", {"product": pr}),
        (f"Apunta {pr}", {"product": pr}),
        (f"Nos hace falta {pr}", {"product": pr}),
        (f"Hay que comprar {pr}", {"product": pr}),
        (f"Ponme {pr} en la lista", {"product": pr}),
        (f"Añade {pr}", {"product": pr}),
        (f"Que no se olvide {pr}", {"product": pr}),
        (f"Acuérdate de comprar {pr}", {"product": pr}),
        (f"Mete {pr} en la lista", {"product": pr}),
    ]

    templates_conversational = [
        (f"Oye, pon {pr} en la lista", {"product": pr}),
        (f"Se nos ha acabado {pr}, apúntalo", {"product": pr}),
        (f"No queda {pr}, ponlo para comprar", {"product": pr}),
        (f"¿Puedes añadir {pr} a la compra?", {"product": pr}),
        (f"Que alguien compre {pr}", {"product": pr}),
        (f"Hay que pillar {pr}", {"product": pr}),
        (f"Cuando vayas al súper compra {pr}", {"product": pr}),
        (f"Para la compra: {pr}", {"product": pr}),
    ]

    all_templates = templates_all + templates_qty + templates_store + templates_minimal + templates_conversational
    text, entities = random.choice(all_templates)
    return maybe_typo(text), entities


def gen_mark_shopping_done():
    pr = random_product()
    q = random_qty()
    s = random_store()

    templates = [
        (f"Ya he comprado {pr}", {"product": pr}),
        (f"Quita {pr} de la lista", {"product": pr}),
        (f"Ya tenemos {pr}", {"product": pr}),
        (f"Tacha {pr}", {"product": pr}),
        (f"Marca como comprado {pr}", {"product": pr}),
        (f"{pr} ya está comprado", {"product": pr}),
        (f"Elimina {pr} de la lista", {"product": pr}),
        (f"Ya pillé {pr}", {"product": pr}),
        (f"Hecho, ya tengo {pr}", {"product": pr}),
        (f"Ya cogí {pr}", {"product": pr}),
        (f"{pr} listo", {"product": pr}),
        (f"Ya compré {pr}", {"product": pr}),
        (f"Borra {pr} de la compra", {"product": pr}),
        (f"Ya lo tengo, {pr}", {"product": pr}),
        (f"{pr} comprado", {"product": pr}),
        (f"Quita {pr}", {"product": pr}),
        (f"Ya está {pr}", {"product": pr}),
        (f"He pillado {pr}", {"product": pr}),
        (f"He comprado {q} de {pr}", {"product": pr}),
        (f"Ya tengo {pr} de {s}", {"product": pr}),
        (f"Pillé {pr} en {s}", {"product": pr}),
        (f"Comprado {pr} en {s}", {"product": pr}),
        (f"Ya he cogido {q} de {pr}", {"product": pr}),
        (f"Oye, ya compré {pr}", {"product": pr}),
        (f"He ido a por {pr} y ya lo tengo", {"product": pr}),
        (f"Listo {pr}, ya lo he comprado", {"product": pr}),
        (f"Quita {pr} que ya lo tengo", {"product": pr}),
        (f"Ya está pillado {pr}", {"product": pr}),
        (f"Acabo de comprar {pr}", {"product": pr}),
        (f"Tacha {pr} que ya lo cogí", {"product": pr}),
        (f"Marca {pr} como hecho", {"product": pr}),
        (f"He pasado por {s} y he pillado {pr}", {"product": pr}),
        (f"Borra {pr}, ya lo compré en {s}", {"product": pr}),
        (f"Ya he ido a comprar {pr}", {"product": pr}),
        (f"Elimina {pr}, comprado", {"product": pr}),
        (f"Hecho lo de {pr}", {"product": pr}),
        (f"Lo de {pr} ya está", {"product": pr}),
        (f"Tengo {pr}, quítalo", {"product": pr}),
        (f"{pr} ya lo he pillado", {"product": pr}),
        (f"Compré {pr} en {s}", {"product": pr}),
    ]

    text, entities = random.choice(templates)
    return maybe_typo(text), entities


def gen_list_shopping():
    # Combinar prefijos + cores + sufijos para más variaciones
    prefixes = [
        "", "Oye, ", "Perdona, ", "Hola, ", "A ver, ", "Dime, ", "Mira, ",
        "Por favor, ", "Porfa, ", "Eh, ", "Venga, ", "Vamos a ver, ",
    ]
    cores = [
        "¿qué hay en la lista de la compra?",
        "¿qué tenemos que comprar?",
        "dime la lista de la compra",
        "enséñame lo que falta por comprar",
        "¿qué necesitamos del súper?",
        "¿qué hay que comprar?",
        "lista de la compra",
        "muéstrame la lista",
        "¿qué falta?",
        "la lista de la compra",
        "¿qué hay apuntado para comprar?",
        "¿tenemos algo en la lista?",
        "¿qué hay pendiente de comprar?",
        "dime qué hay que comprar",
        "¿qué llevo en la lista?",
        "muéstrame qué hay que comprar",
        "ver lista de la compra",
        "¿hay algo en la lista?",
        "¿qué queda por comprar?",
        "la compra",
        "¿qué necesitamos?",
        "lista del súper",
        "¿qué tengo que comprar?",
        "dime la compra",
        "¿qué nos falta?",
        "¿qué hay que pillar?",
        "abre la lista de la compra",
        "¿qué falta por comprar?",
        "¿qué tenemos apuntado?",
        "dame la lista",
        "¿hay algo que comprar?",
        "¿tenemos que ir a comprar algo?",
        "¿necesitamos algo del súper?",
        "cosas que comprar",
        "¿qué llevamos en la lista?",
    ]

    prefix = random.choice(prefixes)
    core = random.choice(cores)
    text = prefix + core
    return maybe_typo(text), {}


def gen_greeting():
    greetings = [
        "Hola", "Buenas", "Buenos días", "Buenas tardes", "Buenas noches",
        "Qué tal", "Hey", "Ey", "Hola buenas", "Hola qué tal",
        "Hola buenos días", "Hola buenas tardes", "Buenas buenas", "Qué hay",
        "Hola hola", "Wenas", "Weenas", "Ei", "Muy buenas",
        "Hola buenas noches", "Qué pasa", "Holi", "Holaa", "Holaaa",
        "Buenaaas", "Hola!", "Buenos dias", "Buenas!", "Ey buenas",
        "Saludos", "Qué tal estás", "Hola qué hay", "Hey buenas",
        "Nas tardes", "Buen día", "Muy buenas tardes", "Eyyy",
        "Que pasa", "K tal", "Hola q tal", "Ola", "Ola buenas", "Wena",
        "Nas", "Bnas", "Buenas a todos", "Hola a todos",
    ]

    suffixes = [
        "", "", "", "", "",  # Más probabilidad de sin sufijo
        ", ¿qué tal?", ", ¿cómo estás?", ", ¿cómo va?",
        "!", "!!", " :)", ", buenas",
        ", ¿qué hay?", ", ¿todo bien?", ", ¿cómo andamos?",
        ", ¿qué se cuenta?", ", ¿qué me cuentas?",
    ]

    text = random.choice(greetings) + random.choice(suffixes)
    return text, {}


def gen_help():
    prefixes = [
        "", "Oye, ", "Perdona, ", "Hola, ", "Eh, ", "Mira, ",
        "Por favor, ", "Porfa, ", "A ver, ", "Disculpa, ",
    ]
    cores = [
        "¿qué puedes hacer?", "ayuda", "¿cómo funciona esto?",
        "¿para qué sirves?", "¿qué opciones tengo?",
        "¿qué sabes hacer?", "¿cómo te uso?",
        "¿qué funciones tienes?", "¿qué haces?",
        "help", "no sé qué hacer", "¿cómo va esto?",
        "¿me puedes ayudar?", "necesito ayuda",
        "¿qué cosas puedes hacer?", "dime qué sabes hacer",
        "¿para qué te puedo usar?", "¿qué comandos hay?",
        "¿qué te puedo pedir?", "¿cómo funciona el chat?",
        "¿qué es esto?", "explícame qué puedes hacer",
        "¿en qué me puedes ayudar?", "dame opciones",
        "no entiendo cómo funciona", "¿qué servicios ofreces?",
        "tutorial", "instrucciones",
        "¿cómo puedo usarte?", "¿qué te puedo decir?",
        "menú", "opciones", "¿qué hay?",
        "no sé cómo funciona esto", "¿me explicas?",
        "¿qué se puede hacer aquí?", "¿tienes instrucciones?",
        "¿cómo te puedo pedir cosas?", "dime las opciones",
        "¿qué puedo hacer aquí?", "¿cómo te funciona?",
        "¿qué cosas haces?", "¿qué me ofreces?",
        "dime cómo va esto", "necesito saber qué haces",
        "¿qué eres?", "¿eres un bot?", "¿eres una inteligencia artificial?",
    ]

    prefix = random.choice(prefixes)
    core = random.choice(cores)
    text = prefix + core
    return text, {}



# ══════════════════════════════════════════════════════════════════════════════
# GENERADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

GENERATORS = {
    "add_medication": (gen_add_medication, 20000),
    "mark_medication_taken": (gen_mark_medication_taken, 15000),
    "list_medications": (gen_list_medications, 10000),
    "check_medication_status": (gen_check_medication_status, 10000),
    "add_reminder": (gen_add_reminder, 20000),
    "list_reminders": (gen_list_reminders, 8000),
    "add_contact": (gen_add_contact, 15000),
    "add_shopping_item": (gen_add_shopping_item, 15000),
    "mark_shopping_done": (gen_mark_shopping_done, 10000),
    "list_shopping": (gen_list_shopping, 5000),
    "greeting": (gen_greeting, 3000),
    "help": (gen_help, 3000),
}


def generate_intent(intent_name, generator_fn, target):
    output_file = f"dataset/{intent_name}.jsonl"
    seen_texts = set()
    examples = []

    attempts = 0
    max_attempts = target * 3  # Evitar bucle infinito

    while len(examples) < target and attempts < max_attempts:
        text, entities = generator_fn()
        attempts += 1

        # Evitar duplicados exactos
        if text.lower() in seen_texts:
            continue

        seen_texts.add(text.lower())
        examples.append({
            "text": text,
            "intent": intent_name,
            "entities": entities,
        })

    # Guardar
    with open(output_file, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return len(examples)


def main():
    os.makedirs("dataset", exist_ok=True)

    print("═══════════════════════════════════════════")
    print("  GENERADOR DE DATASET - PLAMILY CHATBOT")
    print("═══════════════════════════════════════════\n")

    total_target = sum(t for _, t in GENERATORS.values())
    print(f"📊 Intents: {len(GENERATORS)}")
    print(f"🎯 Ejemplos objetivo: {total_target:,}\n")

    total_generated = 0

    for name, (gen_fn, target) in GENERATORS.items():
        print(f"🚀 {name}: generando {target:,} ejemplos...", end=" ", flush=True)
        count = generate_intent(name, gen_fn, target)
        total_generated += count
        status = "✅" if count >= target else f"⚠️ ({count})"
        print(status)

    print(f"\n═══════════════════════════════════════════")
    print(f"✅ TOTAL: {total_generated:,} ejemplos generados")
    print(f"═══════════════════════════════════════════")

    # Crear dataset combinado
    print("\n📦 Creando dataset combinado...")
    all_examples = []
    for name in GENERATORS:
        filepath = f"dataset/{name}.jsonl"
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                all_examples.append(json.loads(line))

    random.shuffle(all_examples)

    # Split: 80% train, 10% val, 10% test
    n = len(all_examples)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    splits = {
        "train": all_examples[:train_end],
        "val": all_examples[train_end:val_end],
        "test": all_examples[val_end:],
    }

    for split_name, split_data in splits.items():
        filepath = f"dataset/{split_name}.jsonl"
        with open(filepath, "w", encoding="utf-8") as f:
            for ex in split_data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {split_name}: {len(split_data):,} ejemplos")

    print("\n✅ Dataset listo en tfm/dataset/")


if __name__ == "__main__":
    main()
