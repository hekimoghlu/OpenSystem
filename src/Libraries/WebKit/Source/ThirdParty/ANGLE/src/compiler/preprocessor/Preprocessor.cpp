/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
// Copyright 2011 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/preprocessor/Preprocessor.h"

#include "common/debug.h"
#include "compiler/preprocessor/DiagnosticsBase.h"
#include "compiler/preprocessor/DirectiveParser.h"
#include "compiler/preprocessor/Macro.h"
#include "compiler/preprocessor/MacroExpander.h"
#include "compiler/preprocessor/Token.h"
#include "compiler/preprocessor/Tokenizer.h"

namespace angle
{

namespace pp
{

struct PreprocessorImpl
{
    Diagnostics *diagnostics;
    MacroSet macroSet;
    Tokenizer tokenizer;
    DirectiveParser directiveParser;
    MacroExpander macroExpander;

    PreprocessorImpl(Diagnostics *diag,
                     DirectiveHandler *directiveHandler,
                     const PreprocessorSettings &settings)
        : diagnostics(diag),
          tokenizer(diag),
          directiveParser(&tokenizer, &macroSet, diag, directiveHandler, settings),
          macroExpander(&directiveParser, &macroSet, diag, settings, false)
    {}
};

Preprocessor::Preprocessor(Diagnostics *diagnostics,
                           DirectiveHandler *directiveHandler,
                           const PreprocessorSettings &settings)
{
    mImpl = new PreprocessorImpl(diagnostics, directiveHandler, settings);
}

Preprocessor::~Preprocessor()
{
    delete mImpl;
}

bool Preprocessor::init(size_t count, const char *const string[], const int length[])
{
    // Add standard pre-defined macros.
    predefineMacro("__LINE__", 0);
    predefineMacro("__FILE__", 0);
    predefineMacro("GL_ES", 1);

    return mImpl->tokenizer.init(count, string, length);
}

void Preprocessor::predefineMacro(const char *name, int value)
{
    PredefineMacro(&mImpl->macroSet, name, value);
}

void Preprocessor::lex(Token *token)
{
    bool validToken = false;
    while (!validToken)
    {
        mImpl->macroExpander.lex(token);
        switch (token->type)
        {
            // We should not be returning internal preprocessing tokens.
            // Convert preprocessing tokens to compiler tokens or report
            // diagnostics.
            case Token::PP_HASH:
                UNREACHABLE();
                break;
            case Token::PP_NUMBER:
                mImpl->diagnostics->report(Diagnostics::PP_INVALID_NUMBER, token->location,
                                           token->text);
                break;
            case Token::PP_OTHER:
                mImpl->diagnostics->report(Diagnostics::PP_INVALID_CHARACTER, token->location,
                                           token->text);
                break;
            default:
                validToken = true;
                break;
        }
    }
}

void Preprocessor::setMaxTokenSize(size_t maxTokenSize)
{
    mImpl->tokenizer.setMaxTokenSize(maxTokenSize);
}

}  // namespace pp

}  // namespace angle
