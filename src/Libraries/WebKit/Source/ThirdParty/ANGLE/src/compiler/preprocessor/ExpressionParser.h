/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#ifndef COMPILER_PREPROCESSOR_EXPRESSIONPARSER_H_
#define COMPILER_PREPROCESSOR_EXPRESSIONPARSER_H_

#include "common/angleutils.h"
#include "compiler/preprocessor/DiagnosticsBase.h"

namespace angle
{

namespace pp
{

class Lexer;
struct Token;

class ExpressionParser : angle::NonCopyable
{
  public:
    struct ErrorSettings
    {
        Diagnostics::ID unexpectedIdentifier;
        bool integerLiteralsMustFit32BitSignedRange;
    };

    ExpressionParser(Lexer *lexer, Diagnostics *diagnostics);

    bool parse(Token *token,
               int *result,
               bool parsePresetToken,
               const ErrorSettings &errorSettings,
               bool *valid);

  private:
    Lexer *mLexer;
    Diagnostics *mDiagnostics;
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_EXPRESSIONPARSER_H_
