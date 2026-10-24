"""Optional bounded extraction schema. / Необязательная ограниченная схема извлечения."""

DEFAULT_LIMITS = {"entities": 16, "relations": 16, "name_chars": 80, "description_chars": 200}


def extraction_schema(ontology, protocol):
    mode = protocol.get("extraction_decoding", "free")
    if mode == "free":
        return None
    if mode != "structured":
        raise ValueError(f"Unknown extraction decoding mode: {mode}")
    limits = {**DEFAULT_LIMITS, **protocol.get("structured_limits", {})}
    if set(limits) != set(DEFAULT_LIMITS) or any(type(v) is not int or v <= 0 for v in limits.values()):
        raise ValueError("structured_limits must contain positive integer limits")
    schema = ontology.full_schema()

    def string(limit):
        return {"type": "string", "minLength": 1, "maxLength": limit}

    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    entity = obj({"entity_name": string(limits["name_chars"]),
                  "entity_type": {"type": "string", "enum": list(schema.entity_types)},
                  "description": string(limits["description_chars"])})
    relation = obj({"source_entity": string(limits["name_chars"]), "target_entity": string(limits["name_chars"]),
                    "relation_type": {"type": "string", "enum": list(schema.relation_types)},
                    "description": string(limits["description_chars"]),
                    "relationship_strength": {"type": "integer", "minimum": 0, "maximum": 5}})
    return obj({"entities": {"type": "array", "items": entity, "maxItems": limits["entities"]},
                "relations": {"type": "array", "items": relation, "maxItems": limits["relations"]}})
