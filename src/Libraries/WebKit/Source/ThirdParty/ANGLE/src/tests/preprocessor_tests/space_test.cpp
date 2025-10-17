/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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

#include <tuple>

#include "PreprocessorTest.h"
#include "compiler/preprocessor/Token.h"

namespace angle
{

class SpaceTest : public PreprocessorTest
{
  protected:
    SpaceTest() : PreprocessorTest(SH_GLES2_SPEC) {}

    void expectSpace(const std::string &str)
    {
        const char *cstr = str.c_str();
        ASSERT_TRUE(mPreprocessor.init(1, &cstr, 0));

        pp::Token token;
        // "foo" is returned after ignoring the whitespace characters.
        mPreprocessor.lex(&token);
        EXPECT_EQ(pp::Token::IDENTIFIER, token.type);
        EXPECT_EQ("foo", token.text);
        // The whitespace character is however recorded with the next token.
        EXPECT_TRUE(token.hasLeadingSpace());
    }
};

// Whitespace characters allowed in GLSL.
// Note that newline characters (\n) will be tested separately.
static const char kSpaceChars[] = {' ', '\t', '\v', '\f'};

// This test fixture tests the processing of a single whitespace character.
// All tests in this fixture are ran with all possible whitespace character
// allowed in GLSL.
class SpaceCharTest : public SpaceTest, public testing::WithParamInterface<char>
{};

TEST_P(SpaceCharTest, SpaceIgnored)
{
    // Construct test string with the whitespace char before "foo".
    std::string str(1, GetParam());
    str.append("foo");

    expectSpace(str);
}

INSTANTIATE_TEST_SUITE_P(SingleSpaceChar, SpaceCharTest, testing::ValuesIn(kSpaceChars));

// This test fixture tests the processing of a string containing consecutive
// whitespace characters. All tests in this fixture are ran with all possible
// combinations of whitespace characters allowed in GLSL.
typedef std::tuple<char, char, char> SpaceStringParams;
class SpaceStringTest : public SpaceTest, public testing::WithParamInterface<SpaceStringParams>
{};

TEST_P(SpaceStringTest, SpaceIgnored)
{
    // Construct test string with the whitespace char before "foo".
    std::string str;
    str.push_back(std::get<0>(GetParam()));
    str.push_back(std::get<1>(GetParam()));
    str.push_back(std::get<2>(GetParam()));
    str.append("foo");

    expectSpace(str);
}

INSTANTIATE_TEST_SUITE_P(SpaceCharCombination,
                         SpaceStringTest,
                         testing::Combine(testing::ValuesIn(kSpaceChars),
                                          testing::ValuesIn(kSpaceChars),
                                          testing::ValuesIn(kSpaceChars)));

// The tests above make sure that the space char is recorded in the
// next token. This test makes sure that a token is not incorrectly marked
// to have leading space.
TEST_F(SpaceTest, LeadingSpace)
{
    const char *str = " foo+ -bar";
    ASSERT_TRUE(mPreprocessor.init(1, &str, 0));

    pp::Token token;
    mPreprocessor.lex(&token);
    EXPECT_EQ(pp::Token::IDENTIFIER, token.type);
    EXPECT_EQ("foo", token.text);
    EXPECT_TRUE(token.hasLeadingSpace());

    mPreprocessor.lex(&token);
    EXPECT_EQ('+', token.type);
    EXPECT_FALSE(token.hasLeadingSpace());

    mPreprocessor.lex(&token);
    EXPECT_EQ('-', token.type);
    EXPECT_TRUE(token.hasLeadingSpace());

    mPreprocessor.lex(&token);
    EXPECT_EQ(pp::Token::IDENTIFIER, token.type);
    EXPECT_EQ("bar", token.text);
    EXPECT_FALSE(token.hasLeadingSpace());
}

}  // namespace angle
