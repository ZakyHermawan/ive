//===----------------------------------------------------------------------===//
//
// This file implements the parser for the Ive language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ive/AST.hpp"
#include "ive/Lexer.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/raw_ostream.h>

#include <memory>

namespace ive {

/// This is a simple recursive parser for the Ive language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &m_lexer);

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule();

private:
  Lexer &m_lexer;

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  std::unique_ptr<ReturnExprAST> parseReturn();

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseNumberExpr();

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr();

  /// Parse a literal struct expression.
  /// structLiteral ::= { (structLiteral | tensorLiteral)+ }
  std::unique_ptr<ExprAST> parseStructLiteralExpr();

  /// parenexpr ::= '(' expression ')'
  std::unique_ptr<ExprAST> parseParenExpr();

  /// Parse a call expression.
  std::unique_ptr<ExprAST> parseCallExpr(llvm::StringRef name,
                                         const Location &loc);

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression ')'
  std::unique_ptr<ExprAST> parseIdentifierExpr();

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> parsePrimary();

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs);

  /// expression::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpression();

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  std::unique_ptr<VarType> parseType();

  /// Parse either a variable declaration or a call expression.
  std::unique_ptr<ExprAST> parseDeclarationOrCallExpr();

  /// Parse a typed variable declaration.
  std::unique_ptr<VarDeclExprAST>
  parseTypedDeclaration(llvm::StringRef typeName, bool requiresInitializer,
                        const Location &loc);

  /// Parse a variable declaration, for either a tensor value or a struct value,
  /// with an optionally required initializer.
  /// decl ::= var identifier [ type ] (= expr)?
  /// decl ::= identifier identifier (= expr)?
  std::unique_ptr<VarDeclExprAST> parseDeclaration(bool requiresInitializer);

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// optionally required initializer.
  /// decl ::= var identifier [ type ] (= expr)?
  std::unique_ptr<VarDeclExprAST> parseVarDeclaration(bool requiresInitializer);

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::unique_ptr<ExprASTList> parseBlock();

  /// prototype ::= def id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  std::unique_ptr<PrototypeAST> parsePrototype();

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> parseDefinition();

  /// Parse a struct definition, we expect a struct initiated with the
  /// `struct` keyword, followed by a block containing a list of variable
  /// declarations.
  ///
  /// definition ::= `struct` identifier `{` decl+ `}`
  std::unique_ptr<StructAST> parseStruct();

  std::unique_ptr<ExprAST> parseIfExpr();

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence();

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = m_lexer.getCurrToken();
    llvm::errs() << "Parse error (" << m_lexer.getLastLocation().line << ", "
                 << m_lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace ive
