/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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

#include "PreprocessorTest.h"
#include "compiler/preprocessor/Token.h"

namespace angle
{

class ErrorTest : public SimplePreprocessorTest
{};

TEST_F(ErrorTest, Empty)
{
    const char *str      = "#error\n";
    const char *expected = "\n";

    using testing::_;
    EXPECT_CALL(mDirectiveHandler, handleError(pp::SourceLocation(0, 1), ""));
    // No error or warning.
    EXPECT_CALL(mDiagnostics, print(_, _, _)).Times(0);

    preprocess(str, expected);
}

TEST_F(ErrorTest, OneTokenMessage)
{
    const char *str      = "#error foo\n";
    const char *expected = "\n";

    using testing::_;
    EXPECT_CALL(mDirectiveHandler, handleError(pp::SourceLocation(0, 1), " foo"));
    // No error or warning.
    EXPECT_CALL(mDiagnostics, print(_, _, _)).Times(0);

    preprocess(str, expected);
}

TEST_F(ErrorTest, TwoTokenMessage)
{
    const char *str      = "#error foo bar\n";
    const char *expected = "\n";

    using testing::_;
    EXPECT_CALL(mDirectiveHandler, handleError(pp::SourceLocation(0, 1), " foo bar"));
    // No error or warning.
    EXPECT_CALL(mDiagnostics, print(_, _, _)).Times(0);

    preprocess(str, expected);
}

TEST_F(ErrorTest, Comments)
{
    const char *str =
        "/*foo*/"
        "#"
        "/*foo*/"
        "error"
        "/*foo*/"
        "foo"
        "/*foo*/"
        "bar"
        "/*foo*/"
        "//foo"
        "\n";
    const char *expected = "\n";

    using testing::_;
    EXPECT_CALL(mDirectiveHandler, handleError(pp::SourceLocation(0, 1), " foo bar"));
    // No error or warning.
    EXPECT_CALL(mDiagnostics, print(_, _, _)).Times(0);

    preprocess(str, expected);
}

TEST_F(ErrorTest, MissingNewline)
{
    const char *str      = "#error foo";
    const char *expected = "";

    using testing::_;
    // Directive successfully parsed.
    EXPECT_CALL(mDirectiveHandler, handleError(pp::SourceLocation(0, 1), " foo"));
    // Error reported about EOF.
    EXPECT_CALL(mDiagnostics, print(pp::Diagnostics::PP_EOF_IN_DIRECTIVE, _, _));

    preprocess(str, expected);
}

}  // namespace angle
