/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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

#ifndef COMPILER_PREPROCESSOR_TOKENIZER_H_
#define COMPILER_PREPROCESSOR_TOKENIZER_H_

#include "common/angleutils.h"
#include "compiler/preprocessor/Input.h"
#include "compiler/preprocessor/Lexer.h"

namespace angle
{

namespace pp
{

class Diagnostics;

class Tokenizer : public Lexer
{
  public:
    struct Context
    {
        Diagnostics *diagnostics;

        Input input;
        // The location where yytext points to. Token location should track
        // scanLoc instead of Input::mReadLoc because they may not be the same
        // if text is buffered up in the scanner input buffer.
        Input::Location scanLoc;

        bool leadingSpace;
        bool lineStart;
    };

    Tokenizer(Diagnostics *diagnostics);
    ~Tokenizer() override;

    bool init(size_t count, const char *const string[], const int length[]);

    void setFileNumber(int file);
    void setLineNumber(int line);
    void setMaxTokenSize(size_t maxTokenSize);

    void lex(Token *token) override;

  private:
    bool initScanner();
    void destroyScanner();

    void *mHandle;         // Scanner handle.
    Context mContext;      // Scanner extra.
    size_t mMaxTokenSize;  // Maximum token size
};

}  // namespace pp

}  // namespace angle

#endif  // COMPILER_PREPROCESSOR_TOKENIZER_H_
