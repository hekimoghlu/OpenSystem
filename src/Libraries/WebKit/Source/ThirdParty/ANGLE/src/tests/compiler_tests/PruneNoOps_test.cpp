/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PruneNoOps_test.cpp:
//   Tests for pruning no-op statements.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

namespace
{

class PruneNoOpsTest : public MatchOutputCodeTest
{
  public:
    PruneNoOpsTest() : MatchOutputCodeTest(GL_FRAGMENT_SHADER, SH_GLSL_COMPATIBILITY_OUTPUT) {}
};

// Test that a switch statement with a constant expression without a matching case is pruned.
TEST_F(PruneNoOpsTest, SwitchStatementWithConstantExpressionNoMatchingCase)
{
    const std::string shaderString = R"(#version 300 es
precision mediump float;
out vec4 color;

void main(void)
{
    switch (10)
    {
        case 0:
            color = vec4(0);
            break;
        case 1:
            color = vec4(1);
            break;
    }
})";
    compile(shaderString);
    ASSERT_TRUE(notFoundInCode("switch"));
    ASSERT_TRUE(notFoundInCode("case"));
}

// Test that a switch statement with a constant expression with a default is not pruned.
TEST_F(PruneNoOpsTest, SwitchStatementWithConstantExpressionWithDefault)
{
    const std::string shaderString = R"(#version 300 es
precision mediump float;
out vec4 color;

void main(void)
{
    switch (10)
    {
        case 0:
            color = vec4(0);
            break;
        case 1:
            color = vec4(1);
            break;
        default:
            color = vec4(0.5);
            break;
    }
})";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("switch"));
    ASSERT_TRUE(foundInCode("case"));
}

// Test that a switch statement with a constant expression with a matching case is not pruned.
TEST_F(PruneNoOpsTest, SwitchStatementWithConstantExpressionWithMatchingCase)
{
    const std::string shaderString = R"(#version 300 es
precision mediump float;
out vec4 color;

void main(void)
{
    switch (10)
    {
        case 0:
            color = vec4(0);
            break;
        case 10:
            color = vec4(1);
            break;
    }
})";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("switch"));
    ASSERT_TRUE(foundInCode("case"));
}

}  // namespace
