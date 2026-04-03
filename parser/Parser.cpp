#include "ive/Parser.hpp"
#include "ive/AST.hpp"
#include "ive/Lexer.hpp"

#include <memory>

namespace ive {

Parser::Parser(Lexer &lexer) : m_lexer(lexer) {}

std::unique_ptr<ModuleAST> Parser::parseModule() {
  m_lexer.getNextToken(); // prime the lexer

  // Parse functions and structs one at a time and accumulate in this vector.
  std::vector<std::unique_ptr<RecordAST>> records;
  while (true) {
    std::unique_ptr<RecordAST> record;
    switch (m_lexer.getCurrToken()) {
    case Token::EndOfFile:
      break;
    case Token::Def:
      record = parseDefinition();
      break;
    case Token::Struct:
      record = parseStruct();
      break;
    default:
      return parseError<ModuleAST>("'def' or 'struct'",
                                   "when parsing top level module records");
    }
    if (!record) {
      break;
    }
    records.push_back(std::move(record));
  }

  // If we didn't reach EOF, there was an error during parsing
  if (m_lexer.getCurrToken() != Token::EndOfFile) {
    return parseError<ModuleAST>("nothing", "at end of module");
  }

  return std::make_unique<ModuleAST>(std::move(records));
}

std::unique_ptr<ReturnExprAST> Parser::parseReturn() {
  auto loc = m_lexer.getLastLocation();
  m_lexer.consume(Token::Return);

  // return takes an optional argument
  std::optional<std::unique_ptr<ExprAST>> expr;
  if (m_lexer.getCurrToken() != Token::Semicolon) {
    expr = parseExpression();
    if (!expr) {
      return nullptr;
    }
  }
  return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
}

std::unique_ptr<ExprAST> Parser::parseNumberExpr() {
  auto loc = m_lexer.getLastLocation();
  auto result =
      std::make_unique<NumberExprAST>(std::move(loc), m_lexer.getValue());
  m_lexer.consume(Token::Number);
  return std::move(result);
}

std::unique_ptr<ExprAST> Parser::parseTensorLiteralExpr() {
  auto loc = m_lexer.getLastLocation();
  m_lexer.consume(Token::SBracketOpen);

  // Hold the list of values at this nesting level.
  std::vector<std::unique_ptr<ExprAST>> values;
  // Hold the dimensions for all the nesting inside this level.
  std::vector<int64_t> dims;
  do {
    // We can have either another nested array or a number literal.
    if (m_lexer.getCurrToken() == Token::SBracketOpen) {
      values.push_back(parseTensorLiteralExpr());
      if (!values.back())
        return nullptr; // parse error in the nested array.
    } else {
      if (m_lexer.getCurrToken() != Token::Number)
        return parseError<ExprAST>("<num> or [", "in literal expression");
      values.push_back(parseNumberExpr());
    }

    // End of this list on ']'
    if (m_lexer.getCurrToken() == Token::SBracketClose)
      break;

    // Elements are separated by a comma.
    if (m_lexer.getCurrToken() != Token::Comma)
      return parseError<ExprAST>("] or ,", "in literal expression");

    m_lexer.getNextToken(); // eat ,
  } while (true);
  if (values.empty())
    return parseError<ExprAST>("<something>", "to fill literal expression");
  m_lexer.getNextToken(); // eat ]

  /// Fill in the dimensions now. First the current nesting level:
  dims.push_back(values.size());

  /// If there is any nested array, process all of them and ensure that
  /// dimensions are uniform.
  if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
        return llvm::isa<LiteralExprAST>(expr.get());
      })) {
    auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
    if (!firstLiteral)
      return parseError<ExprAST>("uniform well-nested dimensions",
                                 "inside literal expression");

    // Append the nested dimensions to the current level
    auto firstDims = firstLiteral->getDims();
    dims.insert(dims.end(), firstDims.begin(), firstDims.end());

    // Sanity check that shape is uniform across all elements of the list.
    for (auto &expr : values) {
      auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
      if (!exprLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");
      if (exprLiteral->getDims() != firstDims)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");
    }
  }
  return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                          std::move(dims));
}

std::unique_ptr<ExprAST> Parser::parseStructLiteralExpr() {
  auto loc = m_lexer.getLastLocation();
  m_lexer.consume(Token::BracketOpen);

  // Hold the list of values.
  std::vector<std::unique_ptr<ExprAST>> values;
  do {
    // We can have either another nested array or a number literal.
    if (m_lexer.getCurrToken() == Token::SBracketOpen) {
      values.push_back(parseTensorLiteralExpr());
      if (!values.back())
        return nullptr;
    } else if (m_lexer.getCurrToken() == Token::Number) {
      values.push_back(parseNumberExpr());
      if (!values.back())
        return nullptr;
    } else {
      if (m_lexer.getCurrToken() != Token::BracketOpen)
        return parseError<ExprAST>("{, [, or number",
                                   "in struct literal expression");
      values.push_back(parseStructLiteralExpr());
    }

    // End of this list on '}'
    if (m_lexer.getCurrToken() == Token::BracketClose)
      break;

    // Elements are separated by a comma.
    if (m_lexer.getCurrToken() != Token::Comma)
      return parseError<ExprAST>("} or ,", "in struct literal expression");

    m_lexer.getNextToken(); // eat ,
  } while (true);
  if (values.empty())
    return parseError<ExprAST>("<something>",
                               "to fill struct literal expression");
  m_lexer.getNextToken(); // eat }

  return std::make_unique<StructLiteralExprAST>(std::move(loc),
                                                std::move(values));
}

std::unique_ptr<ExprAST> Parser::parseParenExpr() {
  m_lexer.getNextToken(); // eat (.
  auto v = parseExpression();
  if (!v)
    return nullptr;

  if (m_lexer.getCurrToken() != Token::ParentheseClose)
    return parseError<ExprAST>(")", "to close expression with parentheses");
  m_lexer.consume(Token::ParentheseClose);
  return v;
}

std::unique_ptr<ExprAST> Parser::parseCallExpr(llvm::StringRef name,
                                               const Location &loc) {
  m_lexer.consume(Token::ParentheseOpen);
  std::vector<std::unique_ptr<ExprAST>> args;
  if (m_lexer.getCurrToken() != Token::ParentheseClose) {
    while (true) {
      if (auto arg = parseExpression()) {
        args.push_back(std::move(arg));
      } else
        return nullptr;

      if (m_lexer.getCurrToken() == Token::ParentheseClose)
        break;

      if (m_lexer.getCurrToken() != Token::Comma)
        return parseError<ExprAST>(", or )", "in argument list");
      m_lexer.getNextToken();
    }
  }
  m_lexer.consume(Token::ParentheseClose);

  // It can be a builtin call to print
  if (name == "print") {
    if (args.size() != 1)
      return parseError<ExprAST>("<single arg>", "as argument to print()");

    return std::make_unique<PrintExprAST>(loc, std::move(args[0]));
  }

  // Call to a user-defined function
  return std::make_unique<CallExprAST>(loc, std::string(name), std::move(args));
}

std::unique_ptr<ExprAST> Parser::parseIdentifierExpr() {
  std::string name(m_lexer.getId());

  auto loc = m_lexer.getLastLocation();
  m_lexer.getNextToken(); // eat identifier.

  if (m_lexer.getCurrToken() != Token::ParentheseOpen) // Simple variable ref.
    return std::make_unique<VariableExprAST>(std::move(loc), name);

  // This is a function call.
  return parseCallExpr(name, loc);
}

std::unique_ptr<ExprAST> Parser::parsePrimary() {
  switch (m_lexer.getCurrToken()) {
  default:
    llvm::errs() << "unknown token '"
                 << static_cast<int>(m_lexer.getCurrToken())
                 << "' when expecting an expression\n";
    return nullptr;
  case Token::Identifier:
    return parseIdentifierExpr();
  case Token::Number:
    return parseNumberExpr();
  case Token::ParentheseOpen:
    return parseParenExpr();
  case Token::SBracketOpen:
    return parseTensorLiteralExpr();
  case Token::BracketOpen:
    return parseStructLiteralExpr();
  case Token::Semicolon:
    return nullptr;
  case Token::BracketClose:
    return nullptr;
  }
}

std::unique_ptr<ExprAST> Parser::parseBinOpRHS(int exprPrec,
                                               std::unique_ptr<ExprAST> lhs) {
  // If this is a binop, find its precedence.
  while (true) {
    int tokPrec = getTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (tokPrec < exprPrec)
      return lhs;

    // Okay, we know this is a binop.
    Token binOp = m_lexer.getCurrToken();
    m_lexer.consume(binOp);
    auto loc = m_lexer.getLastLocation();

    // Parse the primary expression after the binary operator.
    auto rhs = parsePrimary();
    if (!rhs)
      return parseError<ExprAST>("expression", "to complete binary operator");

    // If BinOp binds less tightly with rhs than the operator after rhs, let
    // the pending operator take rhs as its lhs.
    int nextPrec = getTokPrecedence();
    if (tokPrec < nextPrec) {
      rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
      if (!rhs)
        return nullptr;
    }

    // Merge lhs/RHS.
    lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp, std::move(lhs),
                                          std::move(rhs));
  }
}

std::unique_ptr<ExprAST> Parser::parseExpression() {
  auto lhs = parsePrimary();
  if (!lhs)
    return nullptr;

  return parseBinOpRHS(0, std::move(lhs));
}

std::unique_ptr<VarType> Parser::parseType() {
  if (m_lexer.getCurrToken() != Token::Less)
    return parseError<VarType>("<", "to begin type");
  m_lexer.getNextToken(); // eat <

  auto type = std::make_unique<VarType>();

  while (m_lexer.getCurrToken() == Token::Number) {
    type->shape.push_back(m_lexer.getValue());
    m_lexer.getNextToken();
    if (m_lexer.getCurrToken() == Token::Comma)
      m_lexer.getNextToken();
  }

  if (m_lexer.getCurrToken() != Token::Greater)
    return parseError<VarType>(">", "to end type");
  m_lexer.getNextToken(); // eat >
  return type;
}

std::unique_ptr<ExprAST> Parser::parseDeclarationOrCallExpr() {
  auto loc = m_lexer.getLastLocation();
  std::string id(m_lexer.getId());
  m_lexer.consume(Token::Identifier);

  // Check for a call expression.
  if (m_lexer.getCurrToken() == Token::ParentheseOpen)
    return parseCallExpr(id, loc);

  // Otherwise, this is a variable declaration.
  return parseTypedDeclaration(id, /*requiresInitializer=*/true, loc);
}

std::unique_ptr<VarDeclExprAST>
Parser::parseTypedDeclaration(llvm::StringRef typeName,
                              bool requiresInitializer, const Location &loc) {
  // Parse the variable name.
  if (m_lexer.getCurrToken() != Token::Identifier)
    return parseError<VarDeclExprAST>("name", "in variable declaration");
  std::string id(m_lexer.getId());
  m_lexer.getNextToken(); // eat id

  // Parse the initializer.
  std::unique_ptr<ExprAST> expr;
  if (requiresInitializer) {
    if (m_lexer.getCurrToken() != Token::Equal)
      return parseError<VarDeclExprAST>("initializer",
                                        "in variable declaration");
    m_lexer.consume(Token::Equal);
    expr = parseExpression();
  }

  VarType type;
  type.name = std::string(typeName);
  return std::make_unique<VarDeclExprAST>(loc, std::move(id), std::move(type),
                                          std::move(expr));
}

std::unique_ptr<VarDeclExprAST>
Parser::parseDeclaration(bool requiresInitializer) {
  // Check to see if this is a 'var' declaration.
  if (m_lexer.getCurrToken() == Token::Var)
    return parseVarDeclaration(requiresInitializer);

  // Parse the type name.
  if (m_lexer.getCurrToken() != Token::Identifier)
    return parseError<VarDeclExprAST>("type name", "in variable declaration");
  auto loc = m_lexer.getLastLocation();
  std::string typeName(m_lexer.getId());
  m_lexer.getNextToken(); // eat id

  // Parse the rest of the declaration.
  return parseTypedDeclaration(typeName, requiresInitializer, loc);
}

std::unique_ptr<VarDeclExprAST>
Parser::parseVarDeclaration(bool requiresInitializer) {
  if (m_lexer.getCurrToken() != Token::Var)
    return parseError<VarDeclExprAST>("var", "to begin declaration");
  auto loc = m_lexer.getLastLocation();
  m_lexer.getNextToken(); // eat var

  if (m_lexer.getCurrToken() != Token::Identifier)
    return parseError<VarDeclExprAST>("identified", "after 'var' declaration");
  std::string id(m_lexer.getId());
  m_lexer.getNextToken(); // eat id

  std::unique_ptr<VarType> type; // Type is optional, it can be inferred
  if (m_lexer.getCurrToken() == Token::Less) {
    type = parseType();
    if (!type)
      return nullptr;
  }
  if (!type)
    type = std::make_unique<VarType>();

  std::unique_ptr<ExprAST> expr;
  if (requiresInitializer) {
    m_lexer.consume(Token::Equal);
    expr = parseExpression();
  }
  return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                          std::move(*type), std::move(expr));
}

std::unique_ptr<ExprASTList> Parser::parseBlock() {
  if (m_lexer.getCurrToken() != Token::BracketOpen)
    return parseError<ExprASTList>("{", "to begin block");
  m_lexer.consume(Token::BracketOpen);

  auto exprList = std::make_unique<ExprASTList>();

  // Ignore empty expressions: swallow sequences of semicolons.
  while (m_lexer.getCurrToken() == Token::Semicolon)
    m_lexer.consume(Token::Semicolon);

  bool shouldEndsWithSemiColon = true;
  while (m_lexer.getCurrToken() != Token::BracketClose &&
         m_lexer.getCurrToken() != Token::EndOfFile) {
    if (m_lexer.getCurrToken() == Token::Identifier) {
      // Variable declaration or call
      auto expr = parseDeclarationOrCallExpr();
      if (!expr)
        return nullptr;
      exprList->push_back(std::move(expr));
    } else if (m_lexer.getCurrToken() == Token::Var) {
      // Variable declaration
      auto varDecl = parseDeclaration(/*requiresInitializer=*/true);
      if (!varDecl)
        return nullptr;
      exprList->push_back(std::move(varDecl));
    } else if (m_lexer.getCurrToken() == Token::Return) {
      // Return statement
      auto ret = parseReturn();
      if (!ret)
        return nullptr;
      exprList->push_back(std::move(ret));
    } else if (m_lexer.getCurrToken() == Token::If) {
      shouldEndsWithSemiColon = false;
      auto ifExpr = parseIfExpr();
      if (!ifExpr) {
        return nullptr;
      }
      exprList->push_back(std::move(ifExpr));
    } else {
      // General expression
      auto expr = parseExpression();
      if (!expr)
        return nullptr;
      exprList->push_back(std::move(expr));
    }
    // Ensure that elements are separated by a semicolon.
    if (m_lexer.getCurrToken() != Token::Semicolon && shouldEndsWithSemiColon)
      return parseError<ExprASTList>(";", "after expression");

    // Ignore empty expressions: swallow sequences of semicolons.
    while (m_lexer.getCurrToken() == Token::Semicolon)
      m_lexer.consume(Token::Semicolon);
  }

  if (m_lexer.getCurrToken() != Token::BracketClose)
    return parseError<ExprASTList>("}", "to close block");

  m_lexer.consume(Token::BracketClose);
  return exprList;
}

std::unique_ptr<PrototypeAST> Parser::parsePrototype() {
  auto loc = m_lexer.getLastLocation();

  if (m_lexer.getCurrToken() != Token::Def)
    return parseError<PrototypeAST>("def", "in prototype");
  m_lexer.consume(Token::Def);

  if (m_lexer.getCurrToken() != Token::Identifier)
    return parseError<PrototypeAST>("function name", "in prototype");

  std::string fnName(m_lexer.getId());
  m_lexer.consume(Token::Identifier);

  if (m_lexer.getCurrToken() != Token::ParentheseOpen)
    return parseError<PrototypeAST>("(", "in prototype");
  m_lexer.consume(Token::ParentheseOpen);

  std::vector<std::unique_ptr<VarDeclExprAST>> args;
  if (m_lexer.getCurrToken() != Token::ParentheseClose) {
    do {
      VarType type;
      std::string name;

      // Parse either the name of the variable, or its type.
      std::string nameOrType(m_lexer.getId());
      auto loc = m_lexer.getLastLocation();
      m_lexer.consume(Token::Identifier);

      // If the next token is an identifier, we just parsed the type.
      if (m_lexer.getCurrToken() == Token::Identifier) {
        type.name = std::move(nameOrType);

        // Parse the name.
        name = std::string(m_lexer.getId());
        m_lexer.consume(Token::Identifier);
      } else {
        // Otherwise, we just parsed the name.
        name = std::move(nameOrType);
      }

      args.push_back(
          std::make_unique<VarDeclExprAST>(std::move(loc), name, type));
      if (m_lexer.getCurrToken() != Token::Comma)
        break;
      m_lexer.consume(Token::Comma);
      if (m_lexer.getCurrToken() != Token::Identifier)
        return parseError<PrototypeAST>("identifier",
                                        "after ',' in function parameter list");
    } while (true);
  }
  if (m_lexer.getCurrToken() != Token::ParentheseClose)
    return parseError<PrototypeAST>(")", "to end function prototype");

  // success.
  m_lexer.consume(Token::ParentheseClose);
  return std::make_unique<PrototypeAST>(std::move(loc), fnName,
                                        std::move(args));
}

std::unique_ptr<FunctionAST> Parser::parseDefinition() {
  auto proto = parsePrototype();
  if (!proto)
    return nullptr;

  if (auto block = parseBlock())
    return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
  return nullptr;
}

std::unique_ptr<StructAST> Parser::parseStruct() {
  auto loc = m_lexer.getLastLocation();
  m_lexer.consume(Token::Struct);
  if (m_lexer.getCurrToken() != Token::Identifier)
    return parseError<StructAST>("name", "in struct definition");
  std::string name(m_lexer.getId());
  m_lexer.consume(Token::Identifier);

  // Parse: '{'
  if (m_lexer.getCurrToken() != Token::BracketOpen)
    return parseError<StructAST>("{", "in struct definition");
  m_lexer.consume(Token::BracketOpen);

  // Parse: decl+
  std::vector<std::unique_ptr<VarDeclExprAST>> decls;
  do {
    auto decl = parseDeclaration(/*requiresInitializer=*/false);
    if (!decl)
      return nullptr;
    decls.push_back(std::move(decl));

    if (m_lexer.getCurrToken() != Token::Semicolon)
      return parseError<StructAST>(";", "after variable in struct definition");
    m_lexer.consume(Token::Semicolon);
  } while (m_lexer.getCurrToken() != Token::BracketClose);

  // Parse: '}'
  m_lexer.consume(Token::BracketClose);
  return std::make_unique<StructAST>(loc, name, std::move(decls));
}

std::unique_ptr<ExprAST> Parser::parseIfExpr() {
  auto loc = m_lexer.getLastLocation();
  m_lexer.consume(Token::If);

  auto ifExpr = parseExpression();
  if (!ifExpr) {
    return parseError<IfExprAST>("expression", "for if statement");
  }

  auto thenBlock = parseBlock();
  std::unique_ptr<ExprASTList> elseBlock = nullptr;

  if (m_lexer.getCurrToken() == Token::Else) {
    m_lexer.consume(Token::Else);
    elseBlock = parseBlock();
  }

  return std::make_unique<IfExprAST>(
      loc, std::move(ifExpr), std::move(thenBlock), std::move(elseBlock));
}

int Parser::getTokPrecedence() {
  // 1 is lowest precedence.
  switch (m_lexer.getCurrToken()) {
  case Token::Eq:
  case Token::Ne:
    return 8;
  case Token::Lt:
  case Token::Le:
  case Token::Gt:
  case Token::Ge:
    return 10;
  case Token::Minus:
    return 20;
  case Token::Plus:
    return 20;
  case Token::Star:
    return 40;
  case Token::Slash:
    return 40;
  case Token::Dot:
    return 60;
  default:
    return -1;
  }
}

} // namespace ive
