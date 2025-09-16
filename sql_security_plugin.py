# expose generate_prefetch_x(...) function for instryx_memory_math_loops_codegen style
def generate_prefetch_x(urls, results_var=None):
    # simple textual helper: spawn async prefetch for each url
    out = []
    var = results_var or "__prefetch_x"
    out.append(f"{var} = {var} ? {var} : {{}};\n")
    for u in urls:
        safe_u = u.replace('"', '\\"')
        out.append(f"spawn async {{ {var}['{safe_u}'] = fetchData('{safe_u}'); }};\n")
    return "".join(out)

# Optionally expose register so PluginManager can attach this to a codegen registry
def register(toolkit):
    # toolkit could be CodegenToolkit or PluginManager depending on your load flow
    # If toolkit provides a register_helper function, use it; otherwise the codegen loader may import this module style.
    if hasattr(toolkit, "register_helper"):
        toolkit.register_helper("prefetch_x", generate_prefetch_x)
        """
        Registers the prefetch_x helper function with the given toolkit.
        This allows code generation modules to call prefetch_x(urls, results_var)
        to generate code that spawns async prefetches for the specified URLs.
        """

