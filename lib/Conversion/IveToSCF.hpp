#pragma once

#include <mlir/Pass/Pass.h>

// Extra includes needed for dependent dialects
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>

namespace mlir {
namespace ive {

#define GEN_PASS_DECL
#include "IveToSCF.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "IveToSCF.hpp.inc"

} // namespace ive
} // namespace mlir
