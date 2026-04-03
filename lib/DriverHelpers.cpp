#include "ive/DriverHelpers.hpp"
#include "ive/AST.hpp"
#include "ive/Dialect.hpp"
#include "ive/Lexer.hpp"
#include "ive/MLIRGen.hpp"
#include "ive/Parser.hpp"
#include "ive/Passes.hpp"
#include "IveToSCF.hpp"

#include <mlir/Dialect/Affine/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/ModuleTranslation.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>

#include <string>

namespace cl = llvm::cl;

namespace ive {

std::unique_ptr<ive::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  ive::LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  ive::Parser parser(lexer);
  return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module,
             cl::opt<std::string> &inputFileName,
             cl::opt<enum InputType> &inputType) {
  if (inputType != InputType::MLIR &&
      !llvm::StringRef(inputFileName).ends_with(".mlir")) {

    auto moduleAST = parseInputFile(inputFileName);
    if (!moduleAST)
      return 6;

    module = mlirGen(context, *moduleAST);
    return !module ? 1 : 0;
  }

  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFileName);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return -1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);

  if (!module) {
    llvm::errs() << "Error: can't load file " << inputFileName << "\n";
    return 3;
  }

  return 0;
}

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module,
                       cl::opt<std::string> &inputFileName,
                       cl::opt<enum InputType> &inputType,
                       cl::opt<enum Action> &emitAction,
                       cl::opt<bool> &enableOpt) {
  if (int error = loadMLIR(context, module, inputFileName, inputType)) {
    return error;
  }

  mlir::ive::registerIveToSCFPass();
  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return 4;

  bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
  bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

  if (enableOpt || isLoweringToAffine) {
    pm.addPass(mlir::createInlinerPass());

    mlir::OpPassManager &optPM = pm.nest<mlir::ive::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::ive::createShapeInferencePass());
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());
  }

  pm.addPass(mlir::ive::createIveToSCF());
  if (isLoweringToAffine) {
    pm.addPass(mlir::ive::createLowerToAffinePass());

    mlir::OpPassManager &optPM = pm.nest<mlir::ive::FuncOp>();
    optPM.addPass(mlir::createCanonicalizerPass());
    optPM.addPass(mlir::createCSEPass());

    if (enableOpt) {
      optPM.addPass(mlir::affine::createLoopFusionPass());
      optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
    }
  }

  if (isLoweringToLLVM) {
    pm.addPass(mlir::ive::createLowerToLLVMPass());
    pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
  }

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

int dumpAST(cl::opt<std::string> &inputFileName,
            cl::opt<enum InputType> &inputType) {
  if (inputType == InputType::MLIR) {
    llvm::errs() << "Can't dump a Ive AST when the input is MLIR\n";
    return 5;
  }

  auto moduleAST = parseInputFile(inputFileName);
  if (!moduleAST)
    return 1;

  dump(*moduleAST);
  return 0;
}

int dumpLLVMIR(mlir::ModuleOp module, cl::opt<bool> &enableOpt) {
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!tmBuilderOrError) {
    llvm::errs() << "Could not create JITTargetMachineBuilder\n";
    return -1;
  }

  auto tmOrError = tmBuilderOrError->createTargetMachine();
  if (!tmOrError) {
    llvm::errs() << "Could not create TargetMachine\n";
    return -1;
  }
  mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                        tmOrError.get().get());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return -1;
  }
  llvm::errs() << *llvmModule << "\n";
  return 0;
}

int runJit(mlir::ModuleOp module, cl::opt<bool> &enableOpt) {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

} // namespace ive
