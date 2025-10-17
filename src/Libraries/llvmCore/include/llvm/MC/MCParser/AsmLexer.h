/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//===- AsmLexer.h - Lexer for Assembly Files --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class declares the lexer for assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef ASMLEXER_H
#define ASMLEXER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCParser/MCAsmLexer.h"
#include "llvm/Support/DataTypes.h"
#include <string>

namespace llvm {
class MemoryBuffer;
class MCAsmInfo;

/// AsmLexer - Lexer class for assembly files.
class AsmLexer : public MCAsmLexer {
  const MCAsmInfo &MAI;

  const char *CurPtr;
  const MemoryBuffer *CurBuf;
  bool isAtStartOfLine;

  void operator=(const AsmLexer&) LLVM_DELETED_FUNCTION;
  AsmLexer(const AsmLexer&) LLVM_DELETED_FUNCTION;

protected:
  /// LexToken - Read the next token and return its code.
  virtual AsmToken LexToken();

public:
  AsmLexer(const MCAsmInfo &MAI);
  ~AsmLexer();

  void setBuffer(const MemoryBuffer *buf, const char *ptr = NULL);

  virtual StringRef LexUntilEndOfStatement();
  StringRef LexUntilEndOfLine();

  bool isAtStartOfComment(char Char);
  bool isAtStatementSeparator(const char *Ptr);

  const MCAsmInfo &getMAI() const { return MAI; }

private:
  int getNextChar();
  AsmToken ReturnError(const char *Loc, const std::string &Msg);

  AsmToken LexIdentifier();
  AsmToken LexSlash();
  AsmToken LexLineComment();
  AsmToken LexDigit();
  AsmToken LexSingleQuote();
  AsmToken LexQuote();
  AsmToken LexFloatLiteral();
};

} // end namespace llvm

#endif
