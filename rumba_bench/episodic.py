"""Grounded episodic memory before NEREL. / Подтверждённая личная память перед NEREL."""
import json
import re

from .data import canonical, digest

FACT_STATUSES = ("asserted", "plan", "negated", "correction", "deletion", "recommendation", "hypothetical")

FACT_SYSTEM = '''Ты сохраняешь личную эпизодическую память из диалога, а не пишешь энциклопедию.
Извлеки короткие самостоятельные факты о пользователе и упомянутых людях:
действия, покупки, владение, опыт, предпочтения, отношения, количества, даты,
планы, отрицания, изменения, исправления и просьбы забыть информацию.
«Я» и пропущенное подлежащее в репликах user относятся к Пользователю.
Сохраняй, КТО сделал ЧТО, с ЧЕМ, КОГДА и СКОЛЬКО, если это известно.
Сохраняй имена, названия, числа и полные перечисления. Не заменяй факты
определениями предметов. Не добавляй знаний вне сообщений.

Различай роли и модальность:
- «Купила самокат» → Пользователь купила самокат, speaker=user, status=asserted.
- «Хочу купить шлем» → Пользователь хочет купить шлем, speaker=user, status=plan.
  Это НЕ доказательство владения шлемом.
- «Не люблю кофе» → Пользователь не любит кофе, speaker=user, status=negated.
- «Нет, это был чай, а не кофе» → сохранить исправление, status=correction.
- «Забудь адрес» → сохранить просьбу забыть адрес, status=deletion.
- Совет assistant → «Ассистент рекомендовал ...», speaker=assistant,
  status=recommendation. Совет НЕ доказывает действие или предпочтение пользователя.
- Вопрос, предположение и условный пример НЕ превращать в свершившийся факт.

Реплики assistant используй для понимания коротких ответов вроде «да» или
«второй вариант». Конкретные рекомендации (названия, варианты, перечни) сохрани
отдельно с авторством ассистента. Пропускай общие определения и длинные справки.
При исправлении сохраняй изменённый факт с указанием, что именно исправлено.
Относительное время оставляй как сказано: дата сессии хранится отдельно.

Каждый факт должен иметь 1–3 дословные подтверждающие цитаты с message_id.
Цитату копируй ТОЧНО, с исходным регистром и пунктуацией. Для факта пользователя
обязательна цитата user; для совета ассистента — цитата assistant.
Если ответ «да» требует вопроса, процитируй также этот вопрос.
Не повторяй факты. Верни только JSON:
{"facts":[{"statement":"Пользователь ...","speaker":"user","status":"asserted",
"evidence":[{"message_id":0,"quote":"дословная цитата"}]}]}.
Если подтверждённых личных фактов или конкретных рекомендаций нет, верни {"facts":[]}.
'''


def fact_schema(protocol):
    def string(length):
        return {"type": "string", "minLength": 1, "maxLength": length}
    evidence = {"type": "object", "properties": {"message_id": {"type": "integer", "minimum": 0},
                 "quote": string(protocol.get("memory_quote_chars", 400))},
                "required": ["message_id", "quote"], "additionalProperties": False}
    fact = {"type": "object", "properties": {
        "statement": string(protocol.get("memory_statement_chars", 600)),
        "speaker": {"type": "string", "enum": ["user", "assistant"]},
        "status": {"type": "string", "enum": list(FACT_STATUSES)},
        "evidence": {"type": "array", "minItems": 1, "maxItems": 3, "items": evidence}},
        "required": ["statement", "speaker", "status", "evidence"], "additionalProperties": False}
    return {"type": "object", "properties": {"facts": {"type": "array", "maxItems": protocol.get("memory_max_facts", 20),
            "items": fact}}, "required": ["facts"], "additionalProperties": False}


def windows(session, split_text, max_chars=8000):
    """Keep message IDs and roles across splits; retain short preceding context.

    Сохранять номера и роли реплик, включая короткий предыдущий контекст.
    """
    if max_chars < 500:
        raise ValueError("memory_window_chars must be at least 500")
    segments = []
    for i, message in enumerate(session["conversation"]):
        for text in split_text(message["message"], max_chars - 300, 150):
            segments.append({"message_id": i, "role": message["role"], "text": text})
    output, current, size = [], [], 0
    for segment in segments:
        cost = len(segment["text"]) + 60
        if current and size + cost > max_chars:
            output.append(current)
            last = current[-1]
            # A bounded tail helps resolve short replies without silently truncating the new message.
            # Ограниченный хвост помогает понять короткий ответ; новая реплика не обрезается.
            carry = {**last, "text": last["text"][-1000:]}
            current = [carry] if len(carry["text"]) + cost + 60 <= max_chars else []
            size = len(carry["text"]) + 60 if current else 0
        current.append(segment)
        size += cost
    if current:
        output.append(current)
    return output


def fact_prompt(user, session, segments):
    payload = {"user_id": user, "session_datetime": session["session_datetime"],
               "user_statements": [s for s in segments if s["role"] == "user"],
               "assistant_context": [s for s in segments if s["role"] == "assistant"]}
    return [{"role": "system", "content": FACT_SYSTEM}, {"role": "user", "content": canonical(payload)}]


def compact_ws(text):
    return " ".join(text.split())


def parse_facts(text, segments):
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(stripped)
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise ValueError("Expected {facts: [...]}")
    sources = {}
    for segment in segments:
        sources.setdefault(segment["message_id"], []).append(segment)
    accepted, rejected, seen = [], [], set()
    for fact in data["facts"]:
        reason = None
        if not isinstance(fact, dict) or not isinstance(fact.get("statement"), str) or not fact["statement"].strip():
            reason = "invalid_statement"
        elif fact.get("speaker") not in ("user", "assistant") or fact.get("status") not in FACT_STATUSES:
            reason = "invalid_role_or_status"
        elif not isinstance(fact.get("evidence"), list) or not 1 <= len(fact["evidence"]) <= 3:
            reason = "missing_evidence"
        verified, roles = [], set()
        if reason is None:
            for evidence in fact["evidence"]:
                if not isinstance(evidence, dict) or type(evidence.get("message_id")) is not int or not isinstance(evidence.get("quote"), str) or not evidence["quote"].strip():
                    reason = "invalid_evidence"
                    break
                quote = compact_ws(evidence["quote"])
                matches = [s for s in sources.get(evidence["message_id"], []) if quote in compact_ws(s["text"])]
                if not matches:
                    reason = "quote_not_in_visible_source"
                    break
                role = matches[0]["role"]
                roles.add(role)
                verified.append({"message_id": evidence["message_id"], "role": role, "quote": evidence["quote"]})
            if reason is None and fact["speaker"] not in roles:
                reason = "speaker_not_supported_by_evidence"
        if reason:
            rejected.append({"reason": reason, "fact": fact})
            continue
        normalized = {"statement": fact["statement"].strip(), "speaker": fact["speaker"],
                      "status": fact["status"], "evidence": verified}
        key = digest(normalized)
        if key not in seen:
            accepted.append(normalized)
            seen.add(key)
    return accepted, rejected


def nerel_source(facts):
    return "\n".join(f"Факт {i + 1}. {f['statement']} [Автор: {f['speaker']}; модальность: {f['status']}]"
                     for i, f in enumerate(facts))


def citation_repair(rejected, segments, protocol):
    candidates = []
    for item in rejected:
        fact = item.get("fact")
        if not isinstance(fact, dict) or not isinstance(fact.get("statement"), str):
            continue
        if fact.get("speaker") == "user" or (fact.get("speaker") == "assistant" and
            (fact.get("status") == "recommendation" or re.search(r"рекоменд|посовет|предлож|перечисл", fact["statement"], re.I))):
            candidates.append({"candidate_id": len(candidates), "statement": fact["statement"], "speaker": fact["speaker"],
                               "status": fact.get("status", "asserted")})
    if not candidates:
        return None
    system = '''Проверь предложенные факты личной памяти по сообщениям-источникам.
Верни только прямо подтверждённые факты с короткими ДОСЛОВНЫМИ цитатами.
Не меняй statement и speaker кандидата. Если факт не подтверждён, пропусти его.
Нельзя выводить биографию, интерес или действие пользователя только из того,
что он прислал справочную статью или задал вопрос. Общие определения пропускай.
Совет ассистента можно сохранять только с авторством assistant.
Укажи candidate_id. Скопируй цитату точно, обычно достаточно одного короткого
предложения или фразы. Не сокращай цитату многоточиями, не исправляй регистр,
не склеивай разные фрагменты или разные сообщения в одну цитату.
Для двух источников нужны две отдельные записи evidence с их message_id.
Для факта user обязательна цитата user, для assistant — цитата assistant.
Можно уточнить status, сохранив разницу между планом и свершившимся событием.
Верни только JSON {"facts":[{"candidate_id":0,"statement":"исходный statement",
"speaker":"user","status":"asserted","evidence":[{"message_id":0,"quote":"точная цитата"}]}]}.
Если подтверждений нет, верни {"facts":[]}.
'''
    schema = fact_schema(protocol)
    entry = schema["properties"]["facts"]["items"]
    entry["properties"]["candidate_id"] = {"type": "integer", "enum": [c["candidate_id"] for c in candidates]}
    entry["required"].append("candidate_id")
    prompt = [{"role": "system", "content": system},
              {"role": "user", "content": canonical({"candidates": candidates, "messages": segments})}]
    return candidates, prompt, schema


def parse_repaired(text, candidates, segments):
    data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))
    if not isinstance(data, dict) or not isinstance(data.get("facts"), list):
        raise ValueError("Expected repaired facts")
    accepted, seen = [], set()
    for fact in data["facts"]:
        if not isinstance(fact, dict) or type(fact.get("candidate_id")) is not int:
            continue
        index = fact["candidate_id"]
        if not 0 <= index < len(candidates):
            continue
        original = candidates[index]
        if fact.get("statement") != original["statement"] or fact.get("speaker") != original["speaker"]:
            continue
        valid, _ = parse_facts(canonical({"facts": [fact]}), segments)
        for item in valid:
            key = digest(item)
            if key not in seen:
                accepted.append(item)
                seen.add(key)
    return accepted


def fact_documents(user, records):
    """Index selected, verified facts; NEREL artifacts annotate rather than replace them.

    Индексировать выбранные и проверенные факты; NEREL дополняет их, а не заменяет.
    """
    documents, seen = [], set()
    for record in records:
        for i, fact in enumerate(record.get("facts", [])):
            key = (record["session_idx"], digest(fact))
            if key in seen:
                continue
            seen.add(key)
            quotes = "\n".join(f"[{e['message_id']}, {e['role']}] «{e['quote']}»" for e in fact["evidence"])
            text = f"Автор: {fact['speaker']}; модальность: {fact['status']}.\n{fact['statement']}\nПодтверждение:\n{quotes}"
            haystack = compact_ws(text).casefold()
            tags = [e for e in record.get("entities", []) if compact_ws(e["entity_name"]).casefold() in haystack]
            labels = "; ".join(f"{e['entity_name']} ({e['entity_type']})" for e in tags)
            names = {e["entity_name"] for e in tags}
            relations = [r for r in record.get("relations", []) if r["source_entity"] in names and r["target_entity"] in names]
            edges = "; ".join(f"{r['source_entity']} --{r['relation_type']}--> {r['target_entity']}" for r in relations)
            documents.append({"id": f"{user}/{record['session_idx']}/{record['chunk']}/facts/{i}",
                              "session_idx": record["session_idx"], "timestamp": record["timestamp"],
                              "turn_ids": sorted(e["message_id"] for e in fact["evidence"]),
                              "source_turn": max(e["message_id"] for e in fact["evidence"] if e["role"] == fact["speaker"]),
                              "text": text, "index_text": text + ("\nNEREL: " + labels if labels else "") + ("\nСвязи: " + edges if edges else ""),
                              "fact": fact, "nerel_entities": tags, "nerel_relations": relations})
    return documents
