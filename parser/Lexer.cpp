#include "ive/Lexer.hpp"

#include <string>

namespace ive {

Lexer::Lexer(std::string filename)
    : m_lastLocation(
          {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}

Token Lexer::getCurrToken() const { return m_currTok; }

Token Lexer::getNextToken() { return m_currTok = getTok(); }

void Lexer::consume(Token tok) {
  assert(tok == m_currTok && "consume Token mismatch expectation");
  getNextToken();
}

llvm::StringRef Lexer::getId() const {
  assert(m_currTok == tok_identifier);
  return m_identifierStr;
}

double Lexer::getValue() const {
  assert(m_currTok == tok_number);
  return m_numVal;
}

Location Lexer::getLastLocation() const { return m_lastLocation; }

int Lexer::getLine() const { return m_currLineNum; }

int Lexer::getCol() const { return m_currCol; }

int Lexer::getNextChar() {
  // The current line buffer should not be empty unless it is the end of file.
  if (m_currLineBuffer.empty()) {
    return EOF;
  }
  ++m_currCol;
  auto nextchar = m_currLineBuffer.front();
  m_currLineBuffer = m_currLineBuffer.drop_front();

  if (m_currLineBuffer.empty()) {
    m_currLineBuffer = readNextLine();
  }
  if (nextchar == '\n') {
    ++m_currLineNum;
    m_currCol = 0;
  }
  return nextchar;
}

Token Lexer::getTok() {
  // Skip any whitespace.
  while (isspace(m_lastChar)) {
    m_lastChar = Token(getNextChar());
  }

  // Save the current location before reading the token characters.
  m_lastLocation.line = m_currLineNum;
  m_lastLocation.col = m_currCol;

  // Identifier: [a-zA-Z][a-zA-Z0-9_]*
  if (isalpha(m_lastChar)) {
    m_identifierStr = (char)m_lastChar;
    while (isalnum((m_lastChar = Token(getNextChar()))) || m_lastChar == '_') {
      m_identifierStr += (char)m_lastChar;
    }

    if (m_identifierStr == "return") {
      return tok_return;
    }
    if (m_identifierStr == "def") {
      return tok_def;
    }
    if (m_identifierStr == "struct") {
      return tok_struct;
    }
    if (m_identifierStr == "var") {
      return tok_var;
    }
    if (m_identifierStr == "if") {
      return tok_if;
    }
    if (m_identifierStr == "else") {
      return tok_else;
    }

    return tok_identifier;
  }

  // Number: [0-9] ([0-9.])*
  if (isdigit(m_lastChar)) {
    std::string numStr;
    do {
      numStr += m_lastChar;
      m_lastChar = Token(getNextChar());
    } while (isdigit(m_lastChar) || m_lastChar == '.');

    m_numVal = strtod(numStr.c_str(), nullptr);
    return tok_number;
  }

  if (m_lastChar == '#') {
    // Comment until end of line.
    do {
      m_lastChar = Token(getNextChar());
    } while (m_lastChar != EOF && m_lastChar != '\n' && m_lastChar != '\r');

    if (m_lastChar != EOF) {
      return getTok();
    }
  }

  // Check for end of file.  Don't eat the EOF.
  if (m_lastChar == EOF) {
    return tok_eof;
  }

  // Otherwise, just return the character as its ascii value.
  Token thisChar = Token(m_lastChar);
  m_lastChar = Token(getNextChar());
  return thisChar;
}

LexerBuffer::LexerBuffer(const char *begin, const char *end,
                         std::string filename)
    : Lexer(std::move(filename)), m_current(begin), m_end(end) {}

llvm::StringRef LexerBuffer::readNextLine() {
  auto *begin = m_current;
  while (m_current <= m_end && *m_current && *m_current != '\n') {
    ++m_current;
  }
  if (m_current <= m_end && *m_current) {
    ++m_current;
  }
  llvm::StringRef result{begin, static_cast<size_t>(m_current - begin)};
  return result;
}

} // namespace ive
