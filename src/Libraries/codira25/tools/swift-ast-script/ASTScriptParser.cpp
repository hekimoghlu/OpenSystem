/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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

//===--- ASTScriptParser.cpp ----------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
///
/// AST script parsing.
///
//===----------------------------------------------------------------------===//

#include "language/Basic/QuotedString.h"
#include "language/Frontend/Frontend.h"
#include "language/Parse/Lexer.h"
#include "language/Parse/Token.h"
#include "toolchain/Support/MemoryBuffer.h"

#include "ASTScript.h"
#include "ASTScriptConfiguration.h"

using namespace language;
using namespace scripting;

namespace {

class ASTScriptParser {
  ASTScriptConfiguration &Config;
  ASTContext &Context;
  DiagnosticEngine &Diags;
  std::unique_ptr<toolchain::MemoryBuffer> Buffer;
  std::optional<Lexer> TheLexer;
  Token Tok;

public:
  ASTScriptParser(ASTScriptConfiguration &config)
    : Config(config), Context(config.Compiler.getASTContext()),
      Diags(config.Compiler.getDiags()) {

    auto &compiler = config.Compiler;

    // Open the file and give it to the source manager.
    auto bufferOrErr = toolchain::MemoryBuffer::getFile(config.ScriptFile);
    if (!bufferOrErr) {
      toolchain::errs() << "error opening file " << QuotedString(config.ScriptFile)
                   << ": " << bufferOrErr.getError().message() << "\n";
      return;
    }
    auto &sourceManager = compiler.getSourceMgr();
    auto bufferID = sourceManager.addNewSourceBuffer(std::move(*bufferOrErr));

    // Build and prime the lexer.
    TheLexer.emplace(Context.LangOpts, sourceManager, bufferID,
                     &Diags, LexerMode::Codira);
    TheLexer->lex(Tok);
  }

  std::unique_ptr<ASTScript> parseFile() {
    if (!TheLexer) return nullptr;
    return parseTopLevel();
  }

private:
  /***************************************************************************/
  /*** Parsing primitives ****************************************************/
  /***************************************************************************/

  void consume(tok kind) {
    assert(Tok.is(kind));
    TheLexer->lex(Tok);
  }

  bool consumeIf(tok kind) {
    if (Tok.isNot(kind)) return false;
    consume(kind);
    return true;
  }

  bool consumeIfExactly(StringRef literal) {
    if (Tok.isNot(tok::identifier) || Tok.getText() != literal)
      return false;
    consume(tok::identifier);
    return true;
  }

  bool consumeIfIdentifier(StringRef &ident) {
    if (Tok.isNot(tok::identifier)) return false;
    ident = Tok.getText();
    consume(tok::identifier);
    return true;
  }

  std::optional<StringRef> consumeIfIdentifier() {
    StringRef ident;
    return consumeIfIdentifier(ident) ? std::optional<StringRef>(ident)
                                      : std::nullopt;
  }

  /***************************************************************************/
  /*** ASTScript parsing *****************************************************/
  /***************************************************************************/

  std::unique_ptr<ASTScript> parseTopLevel();
};

} // end anonymous namespace

/// ast-script ::= ???
std::unique_ptr<ASTScript> ASTScriptParser::parseTopLevel() {
  return std::unique_ptr<ASTScript>(new ASTScript(Config));
}

std::unique_ptr<ASTScript> ASTScript::parse(ASTScriptConfiguration &config) {
  return ASTScriptParser(config).parseFile();
}
