# instryx_jit_aot_runner.py
# JIT-assisted AOT Execution for Instryx LLVM IR Modules
# Author: Violet Magenta / VACU Technologies
# License: MIT

from instryx_llvm_ir_codegen import InstryxLLVMCodegen
from llvmlite import binding
import ctypes

class InstryxRunner:
    def __init__(self):
        self.codegen = InstryxLLVMCodegen()
        self.engine = None

    def _create_execution_engine(self):
        binding.initialize()
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()

        target = binding.Target.from_default_triple()
        target_machine = target.create_target_machine()
        backing_mod = binding.parse_assembly("")
        engine = binding.create_mcjit_compiler(backing_mod, target_machine)
        return engine

    def _compile_ir(self, llvm_ir: str):
        mod = binding.parse_assembly(llvm_ir)
        mod.verify()
        self.engine.add_module(mod)
        self.engine.finalize_object()
        self.engine.run_static_constructors()

    def run(self, code: str, invoke_main: bool = True):
        llvm_ir = self.codegen.generate(code)
        self.engine = self._create_execution_engine()
        self._compile_ir(llvm_ir)

        if invoke_main:
            func_ptr = self.engine.get_function_address("main")
            cfunc = ctypes.CFUNCTYPE(None)(func_ptr)
            print("🚀 Running Instryx Program...")
            cfunc()

        return llvm_ir


# Test block (can be removed in production)
if __name__ == "__main__":
    runner = InstryxRunner()
    code = """
    func greet(uid) {
        print: "Hello from Instryx IR";
    };

    main() {
        greet(1);
    };
    """
    llvm_ir = runner.run(code)
    print("\n🔬 LLVM IR Output:\n")
    print(llvm_ir)
