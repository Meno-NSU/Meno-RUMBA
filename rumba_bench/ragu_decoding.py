"""Finite-whitespace JSON grammar for the pinned RAGU extraction schemas.

String content stays unrestricted; array lengths and inter-token whitespace are
finite. Unlike JSON-schema maxLength, this does not clip descriptions mid-phrase.
"""
import json


def schema_grammar(schema, whitespace_chars=8):
    """Translate the small schema subset used by TwoStage and its summarizers.

    Fields are emitted in upstream declaration order. Every declared field is
    present; additional properties are disallowed. Unknown schema constructs
    fail explicitly rather than silently weakening the grammar.
    """
    if type(whitespace_chars) is not int or not 0 <= whitespace_chars <= 16:
        raise ValueError("grammar_whitespace_chars must be an integer in [0, 16]")
    rules, references = [], {}

    def literal(value):
        return json.dumps(value, ensure_ascii=False)

    def build(node, name=None):
        if "$ref" in node:
            ref = node["$ref"]
            if not ref.startswith("#/$defs/"):
                raise ValueError(f"Unsupported schema reference: {ref}")
            if ref not in references:
                references[ref] = build(schema["$defs"][ref.split("/")[-1]])
            return references[ref]
        label = name or f"value{len(rules)}"
        position = len(rules)
        rules.append("")
        kind = node.get("type")
        if kind == "object":
            fields = []
            for key, value in node["properties"].items():
                fields.append(f'{literal(json.dumps(key))} ws ":" ws {build(value)}')
            expression = '"{" ws ' + ' ws "," ws '.join(fields) + ' ws "}"'
        elif kind == "array":
            maximum = node.get("maxItems")
            if type(maximum) is not int or maximum <= 0:
                raise ValueError("Grammar arrays require a positive maxItems")
            item = build(node["items"])
            expression = f'"[" ws ({item} (ws "," ws {item}){{0,{maximum - 1}}})? ws "]"'
        elif kind == "string":
            if "maxLength" in node or "pattern" in node or "enum" in node:
                raise ValueError("String restrictions require the bounded JSON-schema mode")
            expression = r'"\"" char* "\""'
        elif kind == "integer" and (node.get("minimum"), node.get("maximum")) == (0, 5):
            expression = ' | '.join(literal(str(value)) for value in range(6))
        else:
            raise ValueError(f"Unsupported RAGU grammar schema: {node}")
        rules[position] = f"{label} ::= {expression}"
        return label

    build(schema, "root")
    rules.extend([
        r'char ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})',
        rf'ws ::= [ \t\n\r]{{0,{whitespace_chars}}}',
    ])
    return "\n".join(rules) + "\n"
