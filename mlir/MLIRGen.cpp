//===----------------------------------------------------------------------===//
//
// This file implements a simple IR generation targeting MLIR from a Module AST
// for the Ive language.
//
//===----------------------------------------------------------------------===//

#include "ive/MLIRGen.hpp"
#include "ive/AST.hpp"
#include "ive/Dialect.hpp"
#include "ive/Lexer.hpp"
#include "ive/Types.hpp"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Value.h>

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Support/ErrorHandling.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::ive;
using namespace ive;

using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using llvm::ScopedHashTableScope;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::Twine;

namespace {

/// Implementation of a simple MLIR emission from the Ive AST.
///
/// This will emit operations that are specific to the Ive language, preserving
/// the semantics of the language and (hopefully) allow to perform accurate
/// analysis and transformation based on these high level semantics.
class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
  }

  /// Public API: convert the AST for a Ive module (source file) to an MLIR
  /// Module operation.
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    // We create an empty MLIR module and codegen functions one at a time and
    // add them to the module.
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (auto &record : moduleAST) {
      if (FunctionAST *funcAST = llvm::dyn_cast<FunctionAST>(record.get())) {
        mlir::ive::FuncOp func = mlirGen(*funcAST);
        if (!func)
          return nullptr;
        functionMap.insert({func.getSymName(), func});
      } else if (StructAST *str = llvm::dyn_cast<StructAST>(record.get())) {
        if (failed(mlirGen(*str)))
          return nullptr;
      } else {
        llvm_unreachable("unknown record type");
      }
    }

    // Verify the module after we have finished constructing it, this will check
    // the structural properties of the IR and invoke any specific verifiers we
    // have on the Ive operations.
    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  /// A "module" matches a Ive source file: containing a list of functions.
  mlir::ModuleOp theModule;

  /// The builder is a helper class to create IR inside a function. The builder
  /// is stateful, in particular it keeps an "insertion point": this is where
  /// the next operations will be introduced.
  mlir::OpBuilder builder;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a function creates a new scope, and the function arguments are
  /// added to the mapping. When the processing of a function is terminated, the
  /// scope is destroyed and the mappings created in this scope are dropped.
  llvm::ScopedHashTable<StringRef, std::pair<mlir::Value, VarDeclExprAST *>>
      symbolTable;
  using SymbolTableScopeT =
      llvm::ScopedHashTableScope<StringRef,
                                 std::pair<mlir::Value, VarDeclExprAST *>>;

  /// A mapping for the functions that have been code generated to MLIR.
  llvm::StringMap<mlir::ive::FuncOp> functionMap;

  /// A mapping for named struct types to the underlying MLIR type and the
  /// original AST node.
  llvm::StringMap<std::pair<mlir::Type, StructAST *>> structMap;

  /// Helper conversion for a Ive AST location to an MLIR location.
  mlir::Location loc(const Location &loc) {
    return mlir::FileLineColLoc::get(builder.getStringAttr(*loc.file), loc.line,
                                     loc.col);
  }

  /// Declare a variable in the current scope, return success if the variable
  /// wasn't declared yet.
  llvm::LogicalResult declare(VarDeclExprAST &var, mlir::Value value) {
    if (symbolTable.count(var.getName()))
      return mlir::failure();
    symbolTable.insert(var.getName(), {value, &var});
    return mlir::success();
  }

  /// Create an MLIR type for the given struct.
  llvm::LogicalResult mlirGen(StructAST &str) {
    if (structMap.count(str.getName()))
      return emitError(loc(str.loc())) << "error: struct type with name `"
                                       << str.getName() << "' already exists";

    auto variables = str.getVariables();
    std::vector<mlir::Type> elementTypes;
    elementTypes.reserve(variables.size());
    for (auto &variable : variables) {
      if (variable->getInitVal())
        return emitError(loc(variable->loc()))
               << "error: variables within a struct definition must not have "
                  "initializers";
      if (!variable->getType().shape.empty())
        return emitError(loc(variable->loc()))
               << "error: variables within a struct definition must not have "
                  "initializers";

      mlir::Type type = getType(variable->getType(), variable->loc());
      if (!type)
        return mlir::failure();
      elementTypes.push_back(type);
    }

    structMap.try_emplace(str.getName(), StructType::get(elementTypes), &str);
    return mlir::success();
  }

  mlir::LogicalResult mlirGen(IfExprAST &ifExpr) {
    auto location = loc(ifExpr.loc());
    auto cond = mlirGen(*ifExpr.getCond());
    if (!cond) {
      return mlir::emitError(
          location, "error: if expression need a conditional statement");
    }

    if (!cond.getType().isInteger(1))
      return mlir::emitError(location)
             << "if condition must have type i1, got " << cond.getType();

    auto thenExpr = ifExpr.getThen();
    auto elseExpr = ifExpr.getElse();

    if (!thenExpr) {
      return mlir::emitError(location,
                             "error: if expression need a then block");
    }

    mlir::ive::IfOp ifOp;
    if (elseExpr && !elseExpr->empty()) {
      ifOp = mlir::ive::IfOp::create(builder, location, cond, true, true);
    } else {
      ifOp = mlir::ive::IfOp::create(builder, location, cond, true, false);
    }

    auto *thenBlock = &ifOp.getThenRegion().back();
    builder.setInsertionPointToStart(thenBlock);
    if (mlir::failed(mlirGen(*ifExpr.getThen()))) {
      return mlir::failure();
    }

    // Only insert ive.yield if needed
    if (thenBlock->empty() ||
        !thenBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      builder.setInsertionPointToEnd(thenBlock);
      mlir::ive::YieldOp::create(builder, location);
    }

    if (elseExpr && !elseExpr->empty()) {
      auto *elseBlock = &ifOp.getElseRegion().back();
      builder.setInsertionPointToStart(elseBlock);
      if (mlir::failed(mlirGen(*elseExpr))) {
        return mlir::failure();
      }
      // Only insert ive.yield if needed
      if (elseBlock->empty() ||
          !elseBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
        builder.setInsertionPointToEnd(elseBlock);
        mlir::ive::YieldOp::create(builder, location);
      }
    }

    return mlir::success();
  }

  static std::optional<double> getConstNumberDoubleExpr(ExprAST *expr) {
    if (auto *number = llvm::dyn_cast_or_null<NumberExprAST>(expr))
      return number->getValueDouble();
    return std::nullopt;
  }

  static bool evalForPredicate(Token op, double lhs, double rhs) {
    switch (op) {
    case Token::Less:
    case Token::Lt:
      return lhs < rhs;
    case Token::Le:
      return lhs <= rhs;
    case Token::Greater:
    case Token::Gt:
      return lhs > rhs;
    case Token::Ge:
      return lhs >= rhs;
    case Token::Eq:
      return lhs == rhs;
    case Token::Ne:
      return lhs != rhs;
    default:
      return false;
    }
  }

  mlir::LogicalResult mlirGen(ForExprAST &forExpr) {
    auto location = loc(forExpr.loc());

    auto *iterVar = forExpr.getIteratorVar();
    if (!iterVar || !iterVar->getInitVal())
      return mlir::emitError(location,
                             "error: malformed for-loop iterator variable");

    auto *cond = llvm::dyn_cast<BinaryExprAST>(forExpr.getCond());
    if (!cond)
      return mlir::emitError(location,
                             "error: for-loop condition must be a comparison");

    auto *condLhs = llvm::dyn_cast<VariableExprAST>(cond->getLHS());
    if (!condLhs || condLhs->getName() != iterVar->getName()) {
      return mlir::emitError(location,
                             "error: for-loop condition lhs must be iterator "
                             "variable");
    }

    // Preserve the existing assignment semantics for constant-bound loops.
    auto initConst = getConstNumberDoubleExpr(iterVar->getInitVal());
    auto upperConst = getConstNumberDoubleExpr(cond->getRHS());
    auto stepConst = getConstNumberDoubleExpr(forExpr.getStep());
    if (initConst && upperConst && stepConst && *stepConst != 0.0) {
      constexpr int64_t kMaxUnrolledIterations = 1000000;
      int64_t iterCount = 0;
      double iter = *initConst;
      while (evalForPredicate(cond->getOp(), iter, *upperConst)) {
        if (iterCount++ >= kMaxUnrolledIterations)
          return mlir::emitError(
              location,
              "error: for-loop exceeded maximum unrolled iterations (1000000)");
        mlir::Value iterValue = ConstantOp::create(builder, location, iter);
        symbolTable.insert(iterVar->getName(), {iterValue, iterVar});
        if (failed(mlirGenNoScope(*forExpr.getBody())))
          return mlir::failure();
        iter += *stepConst;
      }
      return mlir::success();
    }

    mlir::Value lowerBound = mlirGen(*iterVar->getInitVal());
    if (!lowerBound)
      return mlir::failure();

    mlir::Value upperBound = mlirGen(*cond->getRHS());
    if (!upperBound)
      return mlir::failure();

    mlir::Value step = mlirGen(*forExpr.getStep());
    if (!step)
      return mlir::failure();

    StringRef predicate;
    switch (cond->getOp()) {
    case Token::Less:
    case Token::Lt:
      predicate = "lt";
      break;
    case Token::Le:
      predicate = "le";
      break;
    case Token::Greater:
    case Token::Gt:
      predicate = "gt";
      break;
    case Token::Ge:
      predicate = "ge";
      break;
    case Token::Eq:
      predicate = "eq";
      break;
    case Token::Ne:
      predicate = "ne";
      break;
    default:
      return mlir::emitError(location,
                             "error: unsupported for-loop condition predicate");
    }

    auto forOp = ForOp::create(builder, location, lowerBound, upperBound, step,
                               predicate);

    auto &body = forOp.getBody();
    auto *bodyBlock = new mlir::Block();
    body.push_back(bodyBlock);
    bodyBlock->addArgument(lowerBound.getType(), location);

    builder.setInsertionPointToStart(bodyBlock);
    symbolTable.insert(iterVar->getName(),
                       {bodyBlock->getArgument(0), iterVar});
    if (failed(mlirGen(*forExpr.getBody())))
      return mlir::failure();

    if (bodyBlock->empty() ||
        !bodyBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      builder.setInsertionPointToEnd(bodyBlock);
      YieldOp::create(builder, location);
    }

    builder.setInsertionPointAfter(forOp);

    return mlir::success();
  }

  /// Create the prototype for an MLIR function with as many arguments as the
  /// provided Ive AST prototype.
  mlir::ive::FuncOp mlirGen(PrototypeAST &proto) {
    auto location = loc(proto.loc());

    // This is a generic function, the return type will be inferred later.
    llvm::SmallVector<mlir::Type, 4> argTypes;
    argTypes.reserve(proto.getArgs().size());
    for (auto &arg : proto.getArgs()) {
      mlir::Type type = getType(arg->getType(), arg->loc());
      if (!type)
        return nullptr;
      argTypes.push_back(type);
    }
    auto funcType = builder.getFunctionType(argTypes, /*results=*/{});
    return mlir::ive::FuncOp::create(builder, location, proto.getName(),
                                     funcType);
  }

  /// Emit a new function and add it to the MLIR module.
  mlir::ive::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    SymbolTableScopeT varScope(symbolTable);

    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::ive::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();

    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(*std::get<0>(nameValue), std::get<1>(nameValue))))
        return nullptr;
    }

    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);

    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }

    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      ReturnOp::create(builder, loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(
          builder.getFunctionType(function.getFunctionType().getInputs(),
                                  *returnOp.operand_type_begin()));
    }

    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();
    return function;
  }

  /// Return the struct type that is the result of the given expression, or null
  /// if it cannot be inferred.
  StructAST *getStructFor(ExprAST *expr) {
    llvm::StringRef structName;
    if (auto *decl = llvm::dyn_cast<VariableExprAST>(expr)) {
      auto varIt = symbolTable.lookup(decl->getName());
      if (!varIt.first)
        return nullptr;
      structName = varIt.second->getType().name;
    } else if (auto *access = llvm::dyn_cast<BinaryExprAST>(expr)) {
      if (access->getOp() != Token::Dot)
        return nullptr;
      // The name being accessed should be in the RHS.
      auto *name = llvm::dyn_cast<VariableExprAST>(access->getRHS());
      if (!name)
        return nullptr;
      StructAST *parentStruct = getStructFor(access->getLHS());
      if (!parentStruct)
        return nullptr;

      // Get the element within the struct corresponding to the name.
      VarDeclExprAST *decl = nullptr;
      for (auto &var : parentStruct->getVariables()) {
        if (var->getName() == name->getName()) {
          decl = var.get();
          break;
        }
      }
      if (!decl)
        return nullptr;
      structName = decl->getType().name;
    }
    if (structName.empty())
      return nullptr;

    // If the struct name was valid, check for an entry in the struct map.
    auto structIt = structMap.find(structName);
    if (structIt == structMap.end())
      return nullptr;
    return structIt->second.second;
  }

  /// Return the numeric member index of the given struct access expression.
  std::optional<size_t> getMemberIndex(BinaryExprAST &accessOp) {
    assert(accessOp.getOp() == Token::Dot && "expected access operation");

    // Lookup the struct node for the LHS.
    StructAST *structAST = getStructFor(accessOp.getLHS());
    if (!structAST)
      return std::nullopt;

    // Get the name from the RHS.
    VariableExprAST *name = llvm::dyn_cast<VariableExprAST>(accessOp.getRHS());
    if (!name)
      return std::nullopt;

    auto structVars = structAST->getVariables();
    const auto *it = llvm::find_if(structVars, [&](auto &var) {
      return var->getName() == name->getName();
    });
    if (it == structVars.end())
      return std::nullopt;
    return it - structVars.begin();
  }

  static bool isScalarType(mlir::Type type) {
    return type && (type.isInteger(1) || type.isInteger(32) ||
                    type.isInteger(64) || type.isF64());
  }

  static bool isComparison(Token op) {
    return op == Token::Eq || op == Token::Ne || op == Token::Less ||
           op == Token::Greater || op == Token::Lt || op == Token::Le ||
           op == Token::Gt || op == Token::Ge;
  }

  static bool hasFloatLiteral(ExprAST &expr) {
    if (auto *number = dyn_cast<NumberExprAST>(&expr))
      return number->getSpelling().contains('.');
    if (auto *binary = dyn_cast<BinaryExprAST>(&expr))
      return hasFloatLiteral(*binary->getLHS()) ||
             hasFloatLiteral(*binary->getRHS());
    return false;
  }

  // Find an operand context from a variable or a previously defined function.
  mlir::Type getExprType(ExprAST &expr) {
    if (auto *variable = dyn_cast<VariableExprAST>(&expr)) {
      auto value = symbolTable.lookup(variable->getName()).first;
      if (value)
        return value.getType();
    } else if (auto *call = dyn_cast<CallExprAST>(&expr)) {
      auto it = functionMap.find(call->getCallee());
      if (it != functionMap.end()) {
        auto results = it->second.getFunctionType().getResults();
        if (results.size() == 1)
          return results.front();
      }
    } else if (auto *binary = dyn_cast<BinaryExprAST>(&expr)) {
      if (isComparison(binary->getOp()))
        return builder.getI1Type();
      auto type = getExprType(*binary->getLHS());
      if (!type)
        type = getExprType(*binary->getRHS());
      return type;
    }
    return {};
  }

  /// Emit a binary operation
  mlir::Value mlirGen(BinaryExprAST &binop, mlir::Type expectedType = {}) {
    auto operandType = getExprType(*binop.getLHS());
    if (!operandType)
      operandType = getExprType(*binop.getRHS());
    if (!operandType) {
      operandType = expectedType;
      if (expectedType && isComparison(binop.getOp()))
        operandType = hasFloatLiteral(binop) ? mlir::Type(builder.getF64Type())
                                             : mlir::Type(builder.getI64Type());
    }
    if (!isScalarType(operandType))
      operandType = {};
    // First emit the operations for each side of the operation before emitting
    // the operation itself. For example if the expression is `a + foo(a)`
    // 1) First it will visiting the LHS, which will return a reference to the
    //    value holding `a`. This value should have been emitted at declaration
    //    time and registered in the symbol table, so nothing would be
    //    codegen'd. If the value is not in the symbol table, an error has been
    //    emitted and nullptr is returned.
    // 2) Then the RHS is visited (recursively) and a call to `foo` is emitted
    //    and the result value is returned. If an error occurs we get a nullptr
    //    and propagate.
    //
    mlir::Value lhs = mlirGen(*binop.getLHS(), operandType);
    if (!lhs)
      return nullptr;
    auto location = loc(binop.loc());

    // If this is an access operation, handle it immediately.
    if (binop.getOp() == Token::Dot) {
      std::optional<size_t> accessIndex = getMemberIndex(binop);
      if (!accessIndex) {
        emitError(location, "invalid access into struct expression");
        return nullptr;
      }
      return StructAccessOp::create(builder, location, lhs, *accessIndex);
    }

    // Otherwise, this is a normal binary op.
    mlir::Value rhs = mlirGen(*binop.getRHS(), operandType);
    if (!rhs)
      return nullptr;

    if (isScalarType(lhs.getType()) || isScalarType(rhs.getType())) {
      if (lhs.getType() != rhs.getType()) {
        emitError(location, "scalar operands must have the same type");
        return nullptr;
      }
      using namespace mlir::arith;
      bool floating = lhs.getType().isF64();
      switch (binop.getOp()) {
      case Token::Plus:
        if (floating)
          return AddFOp::create(builder, location, lhs, rhs);
        return AddIOp::create(builder, location, lhs, rhs);
      case Token::Minus:
        if (floating)
          return SubFOp::create(builder, location, lhs, rhs);
        return SubIOp::create(builder, location, lhs, rhs);
      case Token::Star:
        if (floating)
          return MulFOp::create(builder, location, lhs, rhs);
        return MulIOp::create(builder, location, lhs, rhs);
      case Token::Slash:
        if (floating)
          return DivFOp::create(builder, location, lhs, rhs);
        if (lhs.getType().isInteger(1))
          return DivUIOp::create(builder, location, lhs, rhs);
        return DivSIOp::create(builder, location, lhs, rhs);
      default:
        break;
      }
      CmpIPredicate ip;
      CmpFPredicate fp;
      switch (binop.getOp()) {
      case Token::Eq:
        ip = CmpIPredicate::eq;
        fp = CmpFPredicate::OEQ;
        break;
      case Token::Ne:
        ip = CmpIPredicate::ne;
        fp = CmpFPredicate::UNE;
        break;
      case Token::Less:
      case Token::Lt:
        ip = CmpIPredicate::slt;
        fp = CmpFPredicate::OLT;
        break;
      case Token::Le:
        ip = CmpIPredicate::sle;
        fp = CmpFPredicate::OLE;
        break;
      case Token::Greater:
      case Token::Gt:
        ip = CmpIPredicate::sgt;
        fp = CmpFPredicate::OGT;
        break;
      case Token::Ge:
        ip = CmpIPredicate::sge;
        fp = CmpFPredicate::OGE;
        break;
      default:
        emitError(location, "unsupported scalar binary operator");
        return nullptr;
      }
      if (floating)
        return CmpFOp::create(builder, location, fp, lhs, rhs);
      if (lhs.getType().isInteger(1)) {
        if (ip == CmpIPredicate::slt)
          ip = CmpIPredicate::ult;
        if (ip == CmpIPredicate::sle)
          ip = CmpIPredicate::ule;
        if (ip == CmpIPredicate::sgt)
          ip = CmpIPredicate::ugt;
        if (ip == CmpIPredicate::sge)
          ip = CmpIPredicate::uge;
      }
      return CmpIOp::create(builder, location, ip, lhs, rhs);
    }

    // Derive the operation name from the binary operator. At the moment we only
    // support '+' and '*'.
    switch (binop.getOp()) {
    case Token::Plus:
      return AddOp::create(builder, location, lhs, rhs);
    case Token::Minus:
      return SubOp::create(builder, location, lhs, rhs);
    case Token::Star:
      return MulOp::create(builder, location, lhs, rhs);
    case Token::Slash:
      return DivOp::create(builder, location, lhs, rhs);
    case Token::Eq:
    case Token::Ne:
    case Token::Less:
    case Token::Greater:
    case Token::Lt:
    case Token::Le:
    case Token::Gt:
    case Token::Ge:
      break;
    default:
      break;
    }

    if (binop.getOp() == Token::Eq)
      return CmpOp::create(builder, location, lhs, rhs, "eq");
    if (binop.getOp() == Token::Ne)
      return CmpOp::create(builder, location, lhs, rhs, "ne");
    if (binop.getOp() == Token::Less)
      return CmpOp::create(builder, location, lhs, rhs, "lt");
    if (binop.getOp() == Token::Greater)
      return CmpOp::create(builder, location, lhs, rhs, "gt");
    if (binop.getOp() == Token::Lt)
      return CmpOp::create(builder, location, lhs, rhs, "lt");
    if (binop.getOp() == Token::Le)
      return CmpOp::create(builder, location, lhs, rhs, "le");
    if (binop.getOp() == Token::Gt)
      return CmpOp::create(builder, location, lhs, rhs, "gt");
    if (binop.getOp() == Token::Ge)
      return CmpOp::create(builder, location, lhs, rhs, "ge");

    emitError(location, "invalid binary operator '")
        << static_cast<int>(binop.getOp()) << "'";
    return nullptr;
  }

  /// This is a reference to a variable in an expression. The variable is
  /// expected to have been declared and so should have a value in the symbol
  /// table, otherwise emit an error and return nullptr.
  mlir::Value mlirGen(VariableExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()).first)
      return variable;

    emitError(loc(expr.loc()), "error: unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  /// Emit a return operation. This will return failure if any generation fails.
  llvm::LogicalResult mlirGen(ReturnExprAST &ret) {
    auto location = loc(ret.loc());

    // 'return' takes an optional expression, handle that case here.
    mlir::Value expr = nullptr;
    if (ret.getExpr().has_value()) {
      if (!(expr = mlirGen(**ret.getExpr())))
        return mlir::failure();
    }

    // Otherwise, this return operation has zero operands.
    ReturnOp::create(builder, location,
                     expr ? ArrayRef(expr) : ArrayRef<mlir::Value>());
    return mlir::success();
  }

  /// Emit a constant for a literal/constant array. It will be emitted as a
  /// flattened array of data in an Attribute attached to a `ive.constant`
  /// operation. See documentation on [Attributes](LangRef.md#attributes) for
  /// more details. Here is an excerpt:
  ///
  ///   Attributes are the mechanism for specifying constant data in MLIR in
  ///   places where a variable is never allowed [...]. They consist of a name
  ///   and a concrete attribute value. The set of expected attributes, their
  ///   structure, and their interpretation are all contextually dependent on
  ///   what they are attached to.
  ///
  /// Example, the source level statement:
  ///   var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  /// will be converted to:
  ///   %0 = "ive.constant"() {value: dense<tensor<2x3xf64>,
  ///     [[1.000000e+00, 2.000000e+00, 3.000000e+00],
  ///      [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> tensor<2x3xf64>
  ///
  mlir::DenseElementsAttr getConstantAttr(LiteralExprAST &lit) {
    // The attribute is a vector with a floating point value per element
    // (number) in the array, see `collectData()` below for more details.
    std::vector<double> data;
    data.reserve(llvm::product_of(lit.getDims()));
    collectData(lit, data);

    // The type of this attribute is tensor of 64-bit floating-point with the
    // shape of the literal.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get(lit.getDims(), elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    return mlir::DenseElementsAttr::get(dataType, llvm::ArrayRef(data));
  }
  mlir::DenseElementsAttr getConstantAttr(NumberExprAST &lit) {
    // The type of this attribute is tensor of 64-bit floating-point with no
    // shape.
    mlir::Type elementType = builder.getF64Type();
    auto dataType = mlir::RankedTensorType::get({}, elementType);

    // This is the actual attribute that holds the list of values for this
    // tensor literal.
    return mlir::DenseElementsAttr::get(dataType,
                                        llvm::ArrayRef(lit.getValueDouble()));
  }
  /// Emit a constant for a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `ive.struct_constant`
  /// operation. This function returns the generated constant, along with the
  /// corresponding struct type.
  std::pair<mlir::ArrayAttr, mlir::Type>
  getConstantAttr(StructLiteralExprAST &lit) {
    std::vector<mlir::Attribute> attrElements;
    std::vector<mlir::Type> typeElements;

    for (auto &var : lit.getValues()) {
      if (auto *number = llvm::dyn_cast<NumberExprAST>(var.get())) {
        attrElements.push_back(getConstantAttr(*number));
        typeElements.push_back(getType(/*shape=*/{}));
      } else if (auto *lit = llvm::dyn_cast<LiteralExprAST>(var.get())) {
        attrElements.push_back(getConstantAttr(*lit));
        typeElements.push_back(getType(/*shape=*/{}));
      } else {
        auto *structLit = llvm::cast<StructLiteralExprAST>(var.get());
        auto attrTypePair = getConstantAttr(*structLit);
        attrElements.push_back(attrTypePair.first);
        typeElements.push_back(attrTypePair.second);
      }
    }
    mlir::ArrayAttr dataAttr = builder.getArrayAttr(attrElements);
    mlir::Type dataType = StructType::get(typeElements);
    return std::make_pair(dataAttr, dataType);
  }

  /// Emit an array literal.
  mlir::Value mlirGen(LiteralExprAST &lit) {
    mlir::Type type = getType(lit.getDims());
    mlir::DenseElementsAttr dataAttribute = getConstantAttr(lit);

    // Build the MLIR op `ive.constant`. This invokes the `ConstantOp::build`
    // method.
    return ConstantOp::create(builder, loc(lit.loc()), type, dataAttribute);
  }

  /// Emit a struct literal. It will be emitted as an array of
  /// other literals in an Attribute attached to a `ive.struct_constant`
  /// operation.
  mlir::Value mlirGen(StructLiteralExprAST &lit) {
    mlir::ArrayAttr dataAttr;
    mlir::Type dataType;
    std::tie(dataAttr, dataType) = getConstantAttr(lit);

    // Build the MLIR op `ive.struct_constant`. This invokes the
    // `StructConstantOp::build` method.
    return StructConstantOp::create(builder, loc(lit.loc()), dataType,
                                    dataAttr);
  }

  /// Recursive helper function to accumulate the data that compose an array
  /// literal. It flattens the nested structure in the supplied vector. For
  /// example with this array:
  ///  [[1, 2], [3, 4]]
  /// we will generate:
  ///  [ 1, 2, 3, 4 ]
  /// Individual numbers are represented as doubles.
  /// Attributes are the way MLIR attaches constant to operations.
  void collectData(ExprAST &expr, std::vector<double> &data) {
    if (auto *lit = dyn_cast<LiteralExprAST>(&expr)) {
      for (auto &value : lit->getValues())
        collectData(*value, data);
      return;
    }

    assert(isa<NumberExprAST>(expr) && "expected literal or number expr");
    data.push_back(cast<NumberExprAST>(expr).getValueDouble());
  }

  /// Emit a call expression. It emits specific operations for the `transpose`
  /// builtin. Other identifiers are assumed to be user-defined functions.
  mlir::Value mlirGen(CallExprAST &call) {
    llvm::StringRef callee = call.getCallee();
    auto location = loc(call.loc());

    mlir::ive::FuncOp calledFunc;
    if (callee != "transpose") {
      auto it = functionMap.find(callee);
      if (it == functionMap.end()) {
        emitError(location)
            << "no defined function found for '" << callee << "'";
        return nullptr;
      }
      calledFunc = it->second;
      if (calledFunc.getFunctionType().getNumInputs() !=
              call.getArgs().size() ||
          calledFunc.getFunctionType().getNumResults() != 1) {
        emitError(location,
                  "function call requires matching arguments and one result");
        return nullptr;
      }
    }

    // A typed parameter provides the context for literal arguments.
    SmallVector<mlir::Value, 4> operands;
    for (auto &expr : call.getArgs()) {
      mlir::Type expectedType;
      if (calledFunc) {
        auto parameterType =
            calledFunc.getFunctionType().getInput(operands.size());
        if (isScalarType(parameterType))
          expectedType = parameterType;
      }
      auto arg = mlirGen(*expr, expectedType);
      if (!arg)
        return nullptr;
      if (expectedType && arg.getType() != expectedType) {
        emitError(location, "scalar function argument type mismatch");
        return nullptr;
      }
      operands.push_back(arg);
    }

    // Builtin calls have their custom operation, meaning this is a
    // straightforward emission.
    if (callee == "transpose") {
      if (call.getArgs().size() != 1) {
        emitError(location, "MLIR codegen encountered an error: ive.transpose "
                            "does not accept multiple arguments");
        return nullptr;
      }
      return TransposeOp::create(builder, location, operands[0]);
    }

    // Otherwise this is a call to a user-defined function. Calls to
    // user-defined functions are mapped to a custom call that takes the callee
    // name as an attribute.
    return GenericCallOp::create(builder, location,
                                 calledFunc.getFunctionType().getResult(0),
                                 callee, operands);
  }

  /// Emit a print expression. It emits specific operations for two builtins:
  /// transpose(x) and print(x).
  llvm::LogicalResult mlirGen(PrintExprAST &call) {
    auto arg = mlirGen(*call.getArg());
    if (!arg)
      return mlir::failure();

    PrintOp::create(builder, loc(call.loc()), arg);
    return mlir::success();
  }

  /// Emit an assignment expression by updating the value bound to a variable.
  mlir::Value mlirGen(AssignExprAST &assign) {
    auto bound = symbolTable.lookup(assign.getName());
    if (!bound.first) {
      emitError(loc(assign.loc()), "error: unknown variable '")
          << assign.getName() << "'";
      return nullptr;
    }

    auto expectedType = isScalarType(bound.first.getType())
                            ? bound.first.getType()
                            : mlir::Type{};
    mlir::Value value = mlirGen(*assign.getValue(), expectedType);
    if (!value)
      return nullptr;

    if (expectedType && value.getType() != expectedType) {
      emitError(loc(assign.loc()), "scalar assignment type mismatch");
      return nullptr;
    }
    symbolTable.insert(assign.getName(), {value, bound.second});
    return value;
  }

  /// Emit a constant for a single number (FIXME: semantic? broadcast?)
  mlir::Value mlirGen(NumberExprAST &num, mlir::Type expectedType = {}) {
    auto location = loc(num.loc());
    if (!expectedType)
      return ConstantOp::create(builder, location, num.getValueDouble());
    if (expectedType.isF64())
      return ScalarConstantOp::create(
          builder, location, builder.getF64FloatAttr(num.getValueDouble()));

    auto integerType = llvm::cast<mlir::IntegerType>(expectedType);
    llvm::StringRef digits = num.getSpelling();
    bool negative = digits.consume_front("-");
    llvm::APInt magnitude;
    if (digits.getAsInteger(10, magnitude)) {
      emitError(location, "integer scalar requires an integer literal");
      return nullptr;
    }
    unsigned width = integerType.getWidth();
    // Use an extra sign bit so range checking never truncates the literal.
    auto integer =
        magnitude.zext(std::max(width + 1, magnitude.getBitWidth() + 1));
    if (negative)
      integer = -integer;
    if ((width == 1 && (negative || magnitude.getActiveBits() > 1)) ||
        (width != 1 && !integer.isSignedIntN(width))) {
      emitError(location) << "integer literal is out of range for "
                          << expectedType;
      return nullptr;
    }
    return ScalarConstantOp::create(
        builder, location,
        builder.getIntegerAttr(integerType, integer.trunc(width)));
  }

  /// Dispatch codegen for the right expression subclass using RTTI.
  mlir::Value mlirGen(ExprAST &expr, mlir::Type expectedType = {}) {
    switch (expr.getKind()) {
    case ive::ExprAST::Expr_BinOp:
      return mlirGen(cast<BinaryExprAST>(expr), expectedType);
    case ive::ExprAST::Expr_Var:
      return mlirGen(cast<VariableExprAST>(expr));
    case ive::ExprAST::Expr_Literal:
      return mlirGen(cast<LiteralExprAST>(expr));
    case ive::ExprAST::Expr_StructLiteral:
      return mlirGen(cast<StructLiteralExprAST>(expr));
    case ive::ExprAST::Expr_Call:
      return mlirGen(cast<CallExprAST>(expr));
    case ive::ExprAST::Expr_Num:
      return mlirGen(cast<NumberExprAST>(expr), expectedType);
    case ive::ExprAST::Expr_Assign:
      return mlirGen(cast<AssignExprAST>(expr));
    default:
      emitError(loc(expr.loc()))
          << "MLIR codegen encountered an unhandled expr kind '"
          << Twine(expr.getKind()) << "'";
      return nullptr;
    }
  }

  /// Handle a variable declaration, we'll codegen the expression that forms the
  /// initializer and record the value in the symbol table before returning it.
  /// Future expressions will be able to reference this variable through symbol
  /// table lookup.
  mlir::Value mlirGen(VarDeclExprAST &vardecl) {
    auto *init = vardecl.getInitVal();
    if (!init) {
      emitError(loc(vardecl.loc()),
                "missing initializer in variable declaration");
      return nullptr;
    }

    auto expectedType = vardecl.getType().typeKind == TypeKind::Tensor
                            ? mlir::Type{}
                            : getType(vardecl.getType(), vardecl.loc());
    mlir::Value value = mlirGen(*init, expectedType);
    if (!value)
      return nullptr;

    // Handle the case where we are initializing a struct value.
    if (expectedType && value.getType() != expectedType) {
      emitError(loc(vardecl.loc()))
          << "scalar initializer has type " << value.getType() << ", expected "
          << expectedType;
      return nullptr;
    }
    VarType varType = vardecl.getType();
    if (!varType.name.empty()) {
      // Check that the initializer type is the same as the variable
      // declaration.
      mlir::Type type = getType(varType, vardecl.loc());
      if (!type)
        return nullptr;
      if (type != value.getType()) {
        emitError(loc(vardecl.loc()))
            << "struct type of initializer is different than the variable "
               "declaration. Got "
            << value.getType() << ", but expected " << type;
        return nullptr;
      }

      // Otherwise, we have the initializer value, but in case the variable was
      // declared with specific shape, we emit a "reshape" operation. It will
      // get optimized out later as needed.
    } else if (!varType.shape.empty()) {
      value = ReshapeOp::create(builder, loc(vardecl.loc()),
                                getType(varType.shape), value);
    }

    // Register the value in the symbol table.
    if (failed(declare(vardecl, value)))
      return nullptr;
    return value;
  }

  /// Codegen a list of expression, return failure if one of them hit an error.
  llvm::LogicalResult mlirGen(ExprASTList &blockAST) {
    SymbolTableScopeT varScope(symbolTable);
    for (auto it = blockAST.begin(); it != blockAST.end(); ++it) {
      auto &expr = *it;
      // Variable declarations
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      // Return statement: only emit if this is the last statement in the block
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get())) {
        // Only emit return if this is the last statement in the block
        if (std::next(it) == blockAST.end())
          return mlirGen(*ret);
        else
          continue;
      }
      // Print statement
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::failure();
        continue;
      }
      // If statement
      if (auto *ifExpr = dyn_cast<IfExprAST>(expr.get())) {
        // Save parent block before entering the if/else regions
        auto *parentBlock = builder.getInsertionBlock();
        if (mlir::failed(mlirGen(*ifExpr)))
          return mlir::failure();
        // Restore insertion point to parent block after if/else
        builder.setInsertionPointToEnd(parentBlock);
        continue;
      }
      // For statement
      if (auto *forExpr = dyn_cast<ForExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*forExpr)))
          return mlir::failure();
        continue;
      }
      // Generic expression dispatch codegen.
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    // Do not emit a terminator (yield/return) here; let the parent (if/else or
    // function) handle it.
    return mlir::success();
  }

  /// Codegen a list of expressions without introducing a new symbol scope.
  /// This is used for loop-unrolled bodies so assignments can carry to the
  /// next iteration and after the loop.
  llvm::LogicalResult mlirGenNoScope(ExprASTList &blockAST) {
    for (auto it = blockAST.begin(); it != blockAST.end(); ++it) {
      auto &expr = *it;
      if (auto *vardecl = dyn_cast<VarDeclExprAST>(expr.get())) {
        if (!mlirGen(*vardecl))
          return mlir::failure();
        continue;
      }
      if (auto *ret = dyn_cast<ReturnExprAST>(expr.get())) {
        if (std::next(it) == blockAST.end())
          return mlirGen(*ret);
        continue;
      }
      if (auto *print = dyn_cast<PrintExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*print)))
          return mlir::failure();
        continue;
      }
      if (auto *ifExpr = dyn_cast<IfExprAST>(expr.get())) {
        auto *parentBlock = builder.getInsertionBlock();
        if (mlir::failed(mlirGen(*ifExpr)))
          return mlir::failure();
        builder.setInsertionPointToEnd(parentBlock);
        continue;
      }
      if (auto *forExpr = dyn_cast<ForExprAST>(expr.get())) {
        if (mlir::failed(mlirGen(*forExpr)))
          return mlir::failure();
        continue;
      }
      if (!mlirGen(*expr))
        return mlir::failure();
    }
    return mlir::success();
  }

  /// Build a tensor type from a list of shape dimensions.
  mlir::Type getType(ArrayRef<int64_t> shape) {
    // If the shape is empty, then this type is unranked.
    if (shape.empty())
      return mlir::UnrankedTensorType::get(builder.getF64Type());

    // Otherwise, we use the given shape.
    return mlir::RankedTensorType::get(shape, builder.getF64Type());
  }

  /// Build an MLIR type from a Ive AST variable type (forward to the generic
  /// getType above for non-struct types).
  mlir::Type getType(const VarType &type, const Location &location) {
    switch (type.typeKind) {
    case TypeKind::I1:
      return builder.getI1Type();
    case TypeKind::I32:
      return builder.getI32Type();
    case TypeKind::I64:
      return builder.getI64Type();
    case TypeKind::F64:
      return builder.getF64Type();
    case TypeKind::Tensor:
      break;
    }
    if (!type.name.empty()) {
      auto it = structMap.find(type.name);
      if (it == structMap.end()) {
        emitError(loc(location))
            << "error: unknown struct type '" << type.name << "'";
        return nullptr;
      }
      return it->second.first;
    }

    return getType(type.shape);
  }
};

} // namespace

namespace ive {

// The public API for codegen.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace ive
