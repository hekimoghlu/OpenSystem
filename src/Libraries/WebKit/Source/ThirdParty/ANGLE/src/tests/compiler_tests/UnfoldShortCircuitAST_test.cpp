/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// UnfoldShortCircuitAST_test.cpp:
//  Tests shader compilation with unfoldShortCircuit workaround on.

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

class UnfoldShortCircuitASTTest : public MatchOutputCodeTest
{
  public:
    UnfoldShortCircuitASTTest() : MatchOutputCodeTest(GL_FRAGMENT_SHADER, SH_GLSL_330_CORE_OUTPUT)
    {
        ShCompileOptions defaultCompileOptions   = {};
        defaultCompileOptions.unfoldShortCircuit = true;
        setDefaultCompileOptions(defaultCompileOptions);
    }
};

// Test unfolding the && operator.
TEST_F(UnfoldShortCircuitASTTest, UnfoldAnd)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision highp float;

        out vec4 color;
        uniform bool b;
        uniform bool b2;

        void main()
        {
            color = vec4(0, 0, 0, 1);
            if (b && b2)
            {
                color = vec4(0, 1, 0, 1);
            }
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("(_ub) ? (_ub2) : (false)"));
}

// Test unfolding the || operator.
TEST_F(UnfoldShortCircuitASTTest, UnfoldOr)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision highp float;

        out vec4 color;
        uniform bool b;
        uniform bool b2;

        void main()
        {
            color = vec4(0, 0, 0, 1);
            if (b || b2)
            {
                color = vec4(0, 1, 0, 1);
            }
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("(_ub) ? (true) : (_ub2)"));
}

// Test unfolding nested && and || operators. Both should be unfolded.
TEST_F(UnfoldShortCircuitASTTest, UnfoldNested)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision highp float;

        out vec4 color;
        uniform bool b;
        uniform bool b2;
        uniform bool b3;

        void main()
        {
            color = vec4(0, 0, 0, 1);
            if (b && (b2 || b3))
            {
                color = vec4(0, 1, 0, 1);
            }
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("(_ub) ? (((_ub2) ? (true) : (_ub3))) : (false)"));
}
