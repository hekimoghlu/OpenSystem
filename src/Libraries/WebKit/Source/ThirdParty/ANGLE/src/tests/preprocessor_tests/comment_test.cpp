/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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

class CommentTest : public SimplePreprocessorTest, public testing::WithParamInterface<const char *>
{};

TEST_P(CommentTest, CommentIgnored)
{
    const char *str = GetParam();

    pp::Token token;
    lexSingleToken(str, &token);
    EXPECT_EQ(pp::Token::LAST, token.type);
}

INSTANTIATE_TEST_SUITE_P(LineComment,
                         CommentTest,
                         testing::Values("//foo\n",  // With newline.
                                         "//foo",    // Without newline.
                                         "//**/",    // Nested block comment.
                                         "////",     // Nested line comment.
                                         "//\""));   // Invalid character.

INSTANTIATE_TEST_SUITE_P(BlockComment,
                         CommentTest,
                         testing::Values("/*foo*/",
                                         "/*foo\n*/",  // With newline.
                                         "/*//*/",     // Nested line comment.
                                         "/*/**/",     // Nested block comment.
                                         "/***/",      // With lone '*'.
                                         "/*\"*/"));   // Invalid character.

class BlockCommentTest : public SimplePreprocessorTest
{};

TEST_F(BlockCommentTest, CommentReplacedWithSpace)
{
    const char *str = "/*foo*/bar";

    pp::Token token;
    lexSingleToken(str, &token);
    EXPECT_EQ(pp::Token::IDENTIFIER, token.type);
    EXPECT_EQ("bar", token.text);
    EXPECT_TRUE(token.hasLeadingSpace());
}

TEST_F(BlockCommentTest, UnterminatedComment)
{
    const char *str = "/*foo";

    using testing::_;
    EXPECT_CALL(mDiagnostics, print(pp::Diagnostics::PP_EOF_IN_COMMENT, _, _));

    preprocess(str);
}

}  // namespace angle
