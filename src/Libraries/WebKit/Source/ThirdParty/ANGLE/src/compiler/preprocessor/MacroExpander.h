/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

//
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_PREPROCESSOR_MACROEXPANDER_H_
#define COMPILER_PREPROCESSOR_MACROEXPANDER_H_

#include <memory>
#include <vector>

#include "compiler/preprocessor/Lexer.h"
#include "compiler/preprocessor/Macro.h"
#include "compiler/preprocessor/Preprocessor.h"
#include "compiler/preprocessor/Token.h"

namespace angle
{

namespace pp
{

class Diagnostics;
struct SourceLocation;

class MacroExpander : public Lexer
{
  public:
    MacroExpander(Lexer *lexer,
                  MacroSet *macroSet,
                  Diagnostics *diagnostics,
                  const PreprocessorSettings &settings,
                  bool parseDefined);
    ~MacroExpander() override;

    void lex(Token *token) override;

  private:
    void getToken(Token *token);
    void ungetToken(const Token &token);
    bool isNextTokenLeftParen();

    bool pushMacro(std::shared_ptr<Macro> macro, const Token &identifier);
    void popMacro();

    bool expandMacro(const Macro &macro, const Token &identifier, std::vector<Token> *replacements);

    typedef std::vector<Token> MacroArg;
    bool collectMacroArgs(const Macro &macro,
                          const Token &identifier,
                          std::vector<MacroArg> *args,
                          SourceLocation *closingParenthesisLocation);
    void replaceMacroParams(const Macro &macro,
                            const std::vector<MacroArg> &args,
                            std::vector<Token> *replacements);

    struct MacroContext
    {
        MacroContext(std::shared_ptr<Macro> macro, std::vector<Token> &&replacements)
            : macro(std::move(macro)), replacements(std::move(replacements))
        {}
        bool empty() const;
        const Token &get();
        void unget();

        std::shared_ptr<Macro> macro;
        std::vector<Token> replacements;
        std::size_t index = 0;
    };

    Lexer *mLexer;
    MacroSet *mMacroSet;
    Diagnostics *mDiagnostics;
    bool mParseDefined;

    std::unique_ptr<Token> mReserveToken;
    std::vector<MacroContext> mContextStack;
    size_t mTotalTokensInContexts;

    PreprocessorSettings mSettings;

    bool mDeferReenablingMacros;
    std::vector<std::shared_ptr<Macro>> mMacrosToReenable;

    class ScopedMacroReenabler;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_MACROEXPANDER_H_
