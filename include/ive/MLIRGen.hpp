#pragma once

namespace mlir {

class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;

} // namespace mlir

namespace ive {

class ModuleAST;

/// Emit IR for the given Ive moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);

} // namespace ive
