# instryx_ciams_ai_engine.py
# CIAMS AI Engine for Instryx Language - Macro Learning & Suggestion System
# Author: Violet Magenta / VACU Technologies
# License: MIT

import json
import re
from collections import defaultdict, Counter

class CIAMSAIEngine:
    def __init__(self):
        self.macro_usage = defaultdict(Counter)
        self.patterns = {
            'inject': re.compile(r'@inject\s+(\w+(\.\w+)?)'),
            'wraptry': re.compile(r'@wraptry\s+(\w+\(.*?\))'),
            'ffi': re.compile(r'@ffi\s+func\s+(\w+)'),
            'memoize': re.compile(r'@memoize\s+(\w+)'),
        }

    def analyze_code(self, code: str, developer_id="default"):
        for macro, pattern in self.patterns.items():
            matches = pattern.findall(code)
            for match in matches:
                self.macro_usage[developer_id][macro] += 1

    def suggest_macros(self, context: str, developer_id="default"):
        suggestions = []
        usage = self.macro_usage[developer_id]
        if "try" in context and "replace" in context:
            suggestions.append("@wraptry")
        if "db." in context or "net." in context:
            suggestions.append("@inject")
        if "cache" in context or "lookup" in context:
            suggestions.append("@memoize")
        if "extern" in context or "header" in context:
            suggestions.append("@ffi")

        suggestions = sorted(set(suggestions), key=lambda m: -usage[m])
        return suggestions[:3]  # return top 3

    def export_profile(self, developer_id="default", path="ciams_profile.json"):
        with open(path, "w") as f:
            json.dump(self.macro_usage[developer_id], f, indent=2)

    def load_profile(self, path="ciams_profile.json", developer_id="default"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
                self.macro_usage[developer_id].update(data)
        except FileNotFoundError:
            pass


# Test block (can be removed in production)
if __name__ == "__main__":
    ai = CIAMSAIEngine()
    sample_code = """
    @inject db.conn;
    @wraptry risky();
    @ffi func external_math(a, b);
    @memoize compute_value;
    """
    ai.analyze_code(sample_code, developer_id="user123")
    context = "risky() and db.conn and replace block"
    suggestions = ai.suggest_macros(context, developer_id="user123")
    print("🤖 Macro Suggestions:", suggestions)
    ai.export_profile("user123")
