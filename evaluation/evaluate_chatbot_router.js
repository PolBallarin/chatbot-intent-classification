"use strict";

const fs = require("fs");
const path = require("path");
const {
  routeChatbotIntentByRules,
} = require("../../functions/chatbot/router");

const casesPath = path.join(__dirname, "chatbot_intent_cases.jsonl");
const lines = fs.readFileSync(casesPath, "utf-8").split(/\r?\n/).filter(Boolean);

const routed = [];
const unrouted = [];
const mismatches = [];

for (const line of lines) {
  const testCase = JSON.parse(line);
  const route = routeChatbotIntentByRules(testCase.text);

  if (!route) {
    unrouted.push(testCase);
    continue;
  }

  routed.push({ testCase, route });
  if (route.intent !== testCase.expected_intent) {
    mismatches.push({ testCase, route });
  }
}

console.log(`Casos totales: ${lines.length}`);
console.log(`Cubiertos por reglas: ${routed.length}`);
console.log(`Delegados al modelo: ${unrouted.length}`);
console.log(`Errores de reglas: ${mismatches.length}`);

if (mismatches.length > 0) {
  console.log("\nMismatches:");
  for (const { testCase, route } of mismatches) {
    console.log(`- "${testCase.text}"`);
    console.log(`  esperado=${testCase.expected_intent} obtenido=${route.intent} razon=${route.reason}`);
  }
  process.exitCode = 1;
}
