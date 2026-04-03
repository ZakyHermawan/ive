//===----------------------------------------------------------------------===//
//
// This file implements a simple Lexer for the Ive language.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <llvm/ADT/StringRef.h>

#include <cstdlib>
#include <memory>
#include <string>

namespace ive {

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum class Token : int {
  Colon = ':',
  Semicolon = ';',
  ParentheseOpen = '(',
  ParentheseClose = ')',
  BracketOpen = '{',
  BracketClose = '}',
  SBracketOpen = '[',
  SBracketClose = ']',
  Comma = ',',
  Plus = '+',
  Minus = '-',
  Star = '*',
  Slash = '/',
  Dot = '.',
  Equal = '=',
  Less = '<',
  Greater = '>',

  EndOfFile = -1,

  // commands
  Return = -2,
  Var = -3,
  Def = -4,
  Struct = -5,
  If = -6,
  Else = -7,

  // primary
  Identifier = -8,
  Number = -9,

  // comparison
  Eq = -10,
  Ne = -11,
  Lt = -12,
  Le = -13,
  Gt = -14,
  Ge = -15,
};

/// The Lexer is an abstract base class providing all the facilities that the
/// Parser expects. It goes through the stream one token at a time and keeps
/// track of the location in the file for debugging purpose.
/// It relies on a subclass to provide a `readNextLine()` method. The subclass
/// can proceed by reading the next line from the standard input or from a
/// memory mapped file.
class Lexer {
public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purpose (attaching a location to a Token).
  Lexer(std::string filename);
  virtual ~Lexer() = default;

  /// Look at the current token in the stream.
  Token getCurrToken() const;

  /// Move to the next token in the stream and return it.
  Token getNextToken();

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok);

  /// Return the current identifier (prereq: getCurToken() == Token::Identifier)
  llvm::StringRef getId() const;

  /// Return the current number (prereq: getCurToken() == Token::Number)
  double getValue() const;

  /// Return the location for the beginning of the current token.
  Location getLastLocation() const;

  // Return the current line in the file.
  int getLine() const;

  // Return the current column in the file.
  int getCol() const;

private:
  /// Delegate to a derived class fetching the next line. Returns an empty
  /// string to signal end of file (EOF). Lines are expected to always finish
  /// with "\n"
  virtual llvm::StringRef readNextLine() = 0;

  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the derived class as
  /// needed.
  int getNextChar();

  ///  Return the next token from standard input.
  Token getTok();

  /// The last token read from the input.
  Token m_currTok = Token::EndOfFile;

  /// Location for `curTok`.
  Location m_lastLocation;

  /// If the current Token is an identifier, this string contains the value.
  std::string m_identifierStr;

  /// If the current Token is a number, this contains the value.
  double m_numVal = 0;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  int m_lastChar = ' ';

  /// Keep track of the current line number in the input stream
  int m_currLineNum = 0;

  /// Keep track of the current column number in the input stream
  int m_currCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`
  llvm::StringRef m_currLineBuffer = "\n";
};

/// A lexer implementation operating on a buffer in memory.
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename);

private:
  /// Provide one line at a time to the Lexer, return an empty string when
  /// reaching the end of the buffer.
  llvm::StringRef readNextLine() override;
  const char *m_current, *m_end;
};

} // namespace ive
