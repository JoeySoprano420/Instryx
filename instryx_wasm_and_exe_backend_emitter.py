# instryx_wasm_and_exe_backend_emitter.py
# Instryx WASM and EXE Backend Emitter (via LLVM toolchain)
# Author: Violet Magenta / VACU Technologies
# License: MIT

import subprocess
import tempfile
import os
from instryx_llvm_ir_codegen import InstryxLLVMCodegen

class InstryxEmitter:
    def __init__(self, output_dir="build"):
        self.codegen = InstryxLLVMCodegen()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def emit(self, code: str, target: str = "exe", output_name: str = "program"):
        llvm_ir = self.codegen.generate(code)
        ir_path = os.path.join(self.output_dir, f"{output_name}.ll")
        bc_path = os.path.join(self.output_dir, f"{output_name}.bc")
        output_path = os.path.join(self.output_dir, output_name + (".exe" if target == "exe" else ".wasm"))

        # Write LLVM IR to file
        with open(ir_path, "w") as f:
            f.write(llvm_ir)

        # Compile IR to bitcode
        subprocess.run(["llvm-as", ir_path, "-o", bc_path], check=True)

        if target == "exe":
            subprocess.run([
                "llc", bc_path, "-filetype=obj", "-o", f"{output_path}.o"
            ], check=True)
            subprocess.run([
                "clang", f"{output_path}.o", "-o", output_path
            ], check=True)
        elif target == "wasm":
            subprocess.run([
                "llc", "-march=wasm32", bc_path, "-o", f"{output_path}.s"
            ], check=True)
            subprocess.run([
                "wasm-ld", f"{output_path}.s", "-o", output_path, "--no-entry", "--export-all"
            ], check=True)
        else:
            raise ValueError("Target must be 'exe' or 'wasm'")

        print(f"✅ Built target: {output_path}")
        return output_path


# Test block (can be removed in production)
if __name__ == "__main__":
    emitter = InstryxEmitter()
    code = """
    func greet(uid) {
        print: "Hello from Instryx LLVM!";
    };

    main() {
        greet(1);
    };
    """
    emitter.emit(code, target="exe", output_name="test_instryx")
