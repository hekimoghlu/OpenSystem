/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 1, 2025.
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

//===-- llvm/MC/MCAsmParserExtension.h - Asm Parser Hooks -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMPARSEREXTENSION_H
#define LLVM_MC_MCASMPARSEREXTENSION_H

#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SMLoc.h"

namespace llvm {
class Twine;

/// \brief Generic interface for extending the MCAsmParser,
/// which is implemented by target and object file assembly parser
/// implementations.
class MCAsmParserExtension {
  MCAsmParserExtension(const MCAsmParserExtension &) LLVM_DELETED_FUNCTION;
  void operator=(const MCAsmParserExtension &) LLVM_DELETED_FUNCTION;

  MCAsmParser *Parser;

protected:
  MCAsmParserExtension();

  // Helper template for implementing static dispatch functions.
  template<typename T, bool (T::*Handler)(StringRef, SMLoc)>
  static bool HandleDirective(MCAsmParserExtension *Target,
                              StringRef Directive,
                              SMLoc DirectiveLoc) {
    T *Obj = static_cast<T*>(Target);
    return (Obj->*Handler)(Directive, DirectiveLoc);
  }

  bool BracketExpressionsSupported;

public:
  virtual ~MCAsmParserExtension();

  /// \brief Initialize the extension for parsing using the given \p Parser.
  /// The extension should use the AsmParser interfaces to register its
  /// parsing routines.
  virtual void Initialize(MCAsmParser &Parser);

  /// @name MCAsmParser Proxy Interfaces
  /// @{

  MCContext &getContext() { return getParser().getContext(); }
  MCAsmLexer &getLexer() { return getParser().getLexer(); }
  MCAsmParser &getParser() { return *Parser; }
  SourceMgr &getSourceManager() { return getParser().getSourceManager(); }
  MCStreamer &getStreamer() { return getParser().getStreamer(); }
  bool Warning(SMLoc L, const Twine &Msg) {
    return getParser().Warning(L, Msg);
  }
  bool Error(SMLoc L, const Twine &Msg) {
    return getParser().Error(L, Msg);
  }
  bool TokError(const Twine &Msg) {
    return getParser().TokError(Msg);
  }

  const AsmToken &Lex() { return getParser().Lex(); }

  const AsmToken &getTok() { return getParser().getTok(); }

  bool HasBracketExpressions() const { return BracketExpressionsSupported; }

  /// @}
};

} // End llvm namespace

#endif
