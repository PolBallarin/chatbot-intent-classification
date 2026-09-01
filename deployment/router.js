"use strict";

const CHATBOT_CONFIDENCE = {
  lowFallback: 0.70,
  modelDirect: 0.85,
  rule: 0.98,
  override: 0.95,
};

const MEDICATION_TERMS = [
  "medicamento", "medicamentos", "medicina", "medicinas", "medicacion",
  "pastilla", "pastillas", "comprimido", "comprimidos", "capsula", "capsulas",
  "jarabe", "gotas", "inhalador", "parche", "pomada", "insulina",
  "paracetamol", "ibuprofeno", "omeprazol", "nolotil", "aspirina",
  "amoxicilina", "lorazepam", "enalapril", "metformina", "simvastatina",
  "levotiroxina", "diclofenaco", "atorvastatina", "bisoprolol",
  "furosemida", "ramipril", "amlodipino", "losartan", "clopidogrel",
  "prednisona", "gabapentina", "tramadol", "pantoprazol", "alprazolam",
];

const SHOPPING_TERMS = [
  "compra", "compras", "lista", "supermercado", "mercadona", "lidl",
  "carrefour", "aldi", "dia", "eroski", "consum", "alcampo", "bonarea",
  "condis", "ahorramas",
];

const ADD_TERMS = [
  "anade", "apunta", "mete", "agrega", "pon", "poner",
  "crea", "crear", "guarda", "guardar",
];

const HELP_PATTERNS = [
  /\bayuda\b/,
  /\bque puedes\b/,
  /\bque haces\b/,
  /\bcomo funciona\b/,
  /\bopciones\b/,
  /\bfuncionalidades\b/,
];

const GREETING_WORDS = new Set(["hola", "buenas", "buenos", "dias", "tardes", "noches", "hey"]);

function stripAccents(text) {
  return String(text || "").normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}

function compactSpaces(text) {
  return text.replace(/\s+/g, " ").trim();
}

function normalizeChatbotText(text) {
  let normalized = stripAccents(text)
      .toLowerCase()
      .replace(/[`´'"]/g, " ")
      .replace(/[^a-z0-9:\s]/g, " ");

  normalized = compactSpaces(normalized)
      .replace(/\bpa\b/g, "para")
      .replace(/\bx\b/g, "por")
      .replace(/\bme tomado\b/g, "me he tomado")
      .replace(/\bsa tomado\b/g, "se ha tomado")
      .replace(/\bsea tomado\b/g, "se ha tomado")
      .replace(/\banadir\b/g, "anade");

  normalized = ` ${normalized} `
      .replace(/\b(oye|eh|vale|porfa|por favor|bueno|pues|anda)\b/g, " ");

  return compactSpaces(normalized);
}

function normalizeChatbotTextForModel(text) {
  return compactSpaces(String(text || "")
      .toLowerCase()
      .replace(/[`´'"]/g, " ")
      .replace(/[¿?¡!,.;()[\]{}]/g, " "));
}

function containsWord(text, word) {
  return new RegExp(`\\b${word}\\b`).test(text);
}

function containsAnyWord(text, words) {
  return words.some((word) => containsWord(text, word));
}

function matchesAny(text, patterns) {
  return patterns.some((pattern) => pattern.test(text));
}

function hasMedicationCue(text) {
  return containsAnyWord(text, MEDICATION_TERMS);
}

function hasShoppingCue(text) {
  return containsAnyWord(text, SHOPPING_TERMS);
}

function hasAddCue(text) {
  return containsAnyWord(text, ADD_TERMS);
}

function hasDoseOrFrequencyCue(text) {
  return /\b\d+\s*(mg|g|ml)\b/.test(text) ||
    /\bcada\s+\d+\s+(hora|horas|dia|dias)\b/.test(text) ||
    /\b(una|dos|media|medio)\s+(pastilla|comprimido|capsula)\b/.test(text) ||
    /\b(a diario|todos los dias|por la manana|por la tarde|por la noche)\b/.test(text);
}

function isGreeting(text) {
  if (!text) return false;
  const words = text.split(/\s+/);
  return words.length <= 3 && words.every((word) => GREETING_WORDS.has(word));
}

function isMedicationStatusQuery(text) {
  return matchesAny(text, [
    /\b(ha|han|se ha|se han)\s+tomado\s+(todo|todas|la medicacion|las pastillas)\b/,
    /\b(todo|todas|alguna|queda|quedan|faltan|pendiente|pendientes)\b.*\b(tomado|medicacion|pastilla|pastillas|medicamento|medicamentos)\b/,
    /\b(comprueba|comprobar|revisa|mira|dime si|estado)\b.*\b(medicacion|pastillas|medicamentos|tomado|tomada)\b/,
  ]);
}

function isMedicationTakenQuestion(rawText, normalizedText) {
  return /[?¿]/.test(String(rawText || "")) &&
    /\b(ha|han|se ha|se han)\s+tomado\b/.test(normalizedText) &&
    hasMedicationCue(normalizedText);
}

function isMedicationTakenCommand(text) {
  return matchesAny(text, [
    /\b(me he tomado|he tomado|ha tomado|han tomado|se ha tomado|se han tomado)\b/,
    /\b(ya|listo|hecho|confirmo)\b.*\b(tomado|tomada)\b/,
    /\bmarca\b.*\b(tomado|tomada)\b/,
    /\bmarcar\b.*\b(tomado|tomada)\b/,
    /\btome\b.*\b(medicamento|medicacion|pastilla|paracetamol|ibuprofeno|omeprazol|nolotil)\b/,
  ]);
}

function isShoppingDoneCommand(text) {
  return matchesAny(text, [
    /\b(he|ha|hemos|ya)\s+(comprado|compre|compramos)\b/,
    /\bmarca\b.*\b(comprado|comprada)\b/,
    /\bmarcar\b.*\b(comprado|comprada)\b/,
  ]);
}

function isListMedications(text) {
  return matchesAny(text, [
    /\b(que|cuales|dime|ver|ensename|muestrame|lista)\b.*\b(medicamentos|medicacion|pastillas)\b/,
    /\b(medicamentos|medicacion|pastillas)\b.*\b(toma|tiene|lleva)\b/,
  ]);
}

function isListShopping(text) {
  return matchesAny(text, [
    /\blista\s+(de\s+la\s+)?compra\b/,
    /\b(que hay|ensename|muestrame|ver|dime)\b.*\blista\b/,
  ]) && !hasMedicationCue(text);
}

function isAddShoppingItem(text) {
  return hasAddCue(text) && hasShoppingCue(text) && !hasMedicationCue(text);
}

function isAddMedication(text) {
  return (hasAddCue(text) || /\b(programa|pauta)\b/.test(text)) &&
    (hasMedicationCue(text) || hasDoseOrFrequencyCue(text));
}

function routeChatbotIntentByRules(text) {
  const normalizedText = normalizeChatbotText(text);

  if (matchesAny(normalizedText, HELP_PATTERNS)) {
    return {intent: "help", confidence: CHATBOT_CONFIDENCE.rule, reason: "help_keywords", normalizedText};
  }
  if (isGreeting(normalizedText)) {
    return {intent: "greeting", confidence: CHATBOT_CONFIDENCE.rule, reason: "greeting_keywords", normalizedText};
  }
  if (isMedicationStatusQuery(normalizedText)) {
    return {intent: "check_medication_status", confidence: CHATBOT_CONFIDENCE.rule, reason: "medication_status_keywords", normalizedText};
  }
  if (isMedicationTakenQuestion(text, normalizedText)) {
    return {intent: "check_medication_status", confidence: CHATBOT_CONFIDENCE.rule, reason: "medication_taken_question", normalizedText};
  }
  if (isMedicationTakenCommand(normalizedText) && hasMedicationCue(normalizedText)) {
    return {intent: "mark_medication_taken", confidence: CHATBOT_CONFIDENCE.rule, reason: "medication_taken_keywords", normalizedText};
  }
  if (isShoppingDoneCommand(normalizedText) && !hasMedicationCue(normalizedText)) {
    return {intent: "mark_shopping_done", confidence: CHATBOT_CONFIDENCE.rule, reason: "shopping_done_keywords", normalizedText};
  }
  if (isListMedications(normalizedText)) {
    return {intent: "list_medications", confidence: CHATBOT_CONFIDENCE.rule, reason: "list_medications_keywords", normalizedText};
  }
  if (isListShopping(normalizedText)) {
    return {intent: "list_shopping", confidence: CHATBOT_CONFIDENCE.rule, reason: "list_shopping_keywords", normalizedText};
  }
  if (isAddShoppingItem(normalizedText)) {
    return {intent: "add_shopping_item", confidence: CHATBOT_CONFIDENCE.rule, reason: "add_shopping_keywords", normalizedText};
  }
  if (isAddMedication(normalizedText)) {
    return {intent: "add_medication", confidence: CHATBOT_CONFIDENCE.rule, reason: "add_medication_keywords", normalizedText};
  }

  return null;
}

function applyChatbotIntentOverride(modelResult, text) {
  const normalizedText = normalizeChatbotText(text);
  const result = {...modelResult, normalizedText, overridden: false, reason: null};

  if (isMedicationStatusQuery(normalizedText) && result.intent === "mark_medication_taken") {
    return {...result, intent: "check_medication_status", confidence: CHATBOT_CONFIDENCE.override, overridden: true, reason: "status_query_over_mark_taken"};
  }
  if (isMedicationTakenCommand(normalizedText) && hasMedicationCue(normalizedText) &&
      (result.intent === "mark_shopping_done" || result.intent === "add_shopping_item" || result.intent === "check_medication_status")) {
    return {...result, intent: "mark_medication_taken", confidence: CHATBOT_CONFIDENCE.override, overridden: true, reason: "medication_taken_over_shopping"};
  }
  if (matchesAny(normalizedText, HELP_PATTERNS) && (result.intent === "greeting" || result.confidence < 0.8)) {
    return {...result, intent: "help", confidence: CHATBOT_CONFIDENCE.override, overridden: true, reason: "help_keywords"};
  }
  if (isAddMedication(normalizedText) && result.intent === "add_shopping_item") {
    return {...result, intent: "add_medication", confidence: CHATBOT_CONFIDENCE.override, overridden: true, reason: "add_medication_over_shopping"};
  }

  return result;
}

module.exports = {
  CHATBOT_CONFIDENCE,
  normalizeChatbotText,
  normalizeChatbotTextForModel,
  routeChatbotIntentByRules,
  applyChatbotIntentOverride,
};
