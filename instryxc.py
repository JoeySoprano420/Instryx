# instryxc.py
# Final CLI Compiler Wrapper for the Instryx Language
# Author: Violet Magenta / VACU Technologies
# License: MIT

import argparse
import sys
import os
from instryx_parser import InstryxParser
from instryx_ast_interpreter import InstryxInterpreter
from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from instryx_jit_aot_runner import InstryxRunner
from instryx_wasm_and_exe_backend_emitter import InstryxEmitter
from instryx_dodecagram_ast_visualizer import DodecagramExporter

def compile_and_run(args):
    if args.run:
        runner = InstryxRunner()
        runner.run(args.file.read())
    elif args.emit == "llvm":
        codegen = InstryxLLVMCodegen()
        print(codegen.generate(args.file.read()))
    elif args.emit in ["exe", "wasm"]:
        emitter = InstryxEmitter()
        emitter.emit(args.file.read(), target=args.emit, output_name=args.output)
    elif args.emit == "ast":
        parser = InstryxParser()
        ast = parser.parse(args.file.read())
        print(ast)
    elif args.emit == "visual":
        viz = DodecagramExporter()
        viz.parse_code(args.file.read())
        viz.export_to_graphviz(f"{args.output}_ast")
    elif args.emit == "json":
        viz = DodecagramExporter()
        viz.parse_code(args.file.read())
        viz.export_to_json(f"{args.output}_ast.json")
    elif args.emit == "interpret":
        interpreter = InstryxInterpreter()
        interpreter.interpret(args.file.read())
    else:
        print("Unknown emit mode:", args.emit)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="🧠 Instryx CLI Compiler")
    parser.add_argument("file", type=argparse.FileType("r"), help="Instryx source file (.ix)")
    parser.add_argument("-o", "--output", type=str, default="program", help="Output file name prefix")
    parser.add_argument("--emit", type=str, choices=["llvm", "exe", "wasm", "ast", "visual", "json", "interpret"],
                        help="What to emit: LLVM IR, binary, AST, visual, etc.")
    parser.add_argument("--run", action="store_true", help="JIT compile and run the code")

    args = parser.parse_args()
    compile_and_run(args)

if __name__ == "__main__":
    main()
