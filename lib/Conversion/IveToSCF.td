#ifndef LIB_CONVERSION_IVETOSCF_IVETOSCF_TD_
#define LIB_CONVERSION_IVETOSCF_IVETOSCF_TD_

include "mlir/Pass/PassBase.td"

def IveToSCF : Pass<"ive-to-scf"> {
  let summary = "Lower `ive` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `ive` dialect to SCF Dialect
  }];

  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::ive::IveDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_CONVERSION_IVETOSCF_IVETOSCF_TD_
