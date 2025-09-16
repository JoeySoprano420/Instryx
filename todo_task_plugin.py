# register(assistant) style plugin that flags TODO comments as suggestions
from ciams.ai_engine import Suggestion

def rule_todos(source: str, filename=None):
    suggestions = []
    for m in __import__('re').finditer(r"//\s*TODO[: ]?(.*)$|#\s*TODO[: ]?(.*)$", source, __import__('re').M):
        txt = (m.group(1) or m.group(2) or "").strip()
        snippet = source[max(0, m.start()-60): m.end()+60].splitlines()[0].strip()
        suggestions.append(Suggestion("note", [txt], f"TODO found: {txt}", 0.2, snippet, (m.start(), m.end())))
    return suggestions

def register(assistant):
    assistant.register_rule(rule_todos)
    def unregister(assistant):
        # optional: if your PluginManager supports unregister, implement removal logic
        pass
    assistant.unregister_rule(rule_todos)
    # register(assistant) style plugin that flags TODO comments as suggestions
    """
    Registers the rule_todos function with the assistant to flag TODO comments in the source code.
    """
