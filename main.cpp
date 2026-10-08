#include "ive/Dialect.hpp"
#include "ive/DriverHelpers.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Extensions/AllExtensions.h>
#include <mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <string>

using namespace ive;
namespace cl = llvm::cl;

int main(int argc, char **argv) {
  cl::opt<std::string> inputFileName(cl::Positional,
                                     cl::desc("<input ive file>"),
                                     cl::init("-"), cl::value_desc("filename"));

  cl::opt<enum InputType> inputType(
      "x", cl::init(Ive), cl::desc("Decided the kind of output desired"),
      cl::values(
          clEnumValN(Ive, "ive", "load the input file as an Ive source.")),
      cl::values(
          clEnumValN(MLIR, "mlir", "load the input file as an MLIR file")));

  cl::opt<enum Action> emitAction(
      "emit", cl::desc("Select the kind of output desired"),
      cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
      cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
      cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                            "output the MLIR dump after affine lowering")),
      cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                            "output the MLIR dump after llvm lowering")),
      cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
      cl::values(
          clEnumValN(RunJIT, "jit",
                     "JIT the code and run it by invoking the main function")));

  cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "ive compiler\n");

  if (emitAction == DumpAST) {
    return dumpAST(inputFileName, inputType);
  }

  // If we aren't dumping the AST, then we are compiling with/to MLIR.
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect>();
  mlir::func::registerAllExtensions(registry);
  mlir::LLVM::registerInlinerInterface(registry);

  mlir::MLIRContext context(registry);
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::ive::IveDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module;
  if (int error = loadAndProcessMLIR(context, module, inputFileName, inputType,
                                     emitAction, enableOpt)) {
    return error;
  }

  // If we aren't exporting to non-mlir, then we are done.
  bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
  if (isOutputingMLIR) {
    module->dump();
    return 0;
  }

  // Check to see if we are compiling to LLVM IR.
  if (emitAction == Action::DumpLLVMIR) {
    return dumpLLVMIR(*module, enableOpt);
  }

  // Otherwise, we must be running the jit.
  if (emitAction == RunJIT) {
    return runJit(*module, enableOpt);
  }

  llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
  return -1;
}
