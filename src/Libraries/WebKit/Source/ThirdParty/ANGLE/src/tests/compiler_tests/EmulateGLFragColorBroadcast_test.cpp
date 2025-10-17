/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// EmulateGLFragColorBroadcast_test.cpp:
//   Tests for gl_FragColor broadcast behavior emulation.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

namespace
{

const int kMaxDrawBuffers = 2;

class EmulateGLFragColorBroadcastTest : public MatchOutputCodeTest
{
  public:
    EmulateGLFragColorBroadcastTest()
        : MatchOutputCodeTest(GL_FRAGMENT_SHADER, SH_GLSL_COMPATIBILITY_OUTPUT)
    {
        getResources()->MaxDrawBuffers   = kMaxDrawBuffers;
        getResources()->EXT_draw_buffers = 1;
    }
};

// Verifies that without explicitly enabling GL_EXT_draw_buffers extension
// in the shader, no broadcast emulation.
TEST_F(EmulateGLFragColorBroadcastTest, FragColorNoBroadcast)
{
    const std::string shaderString =
        "void main()\n"
        "{\n"
        "    gl_FragColor = vec4(1, 0, 0, 0);\n"
        "}\n";
    compile(shaderString);
    EXPECT_TRUE(foundInCode("gl_FragColor"));
    EXPECT_FALSE(foundInCode("gl_FragData[0]"));
    EXPECT_FALSE(foundInCode("gl_FragData[1]"));
}

// Verifies that with explicitly enabling GL_EXT_draw_buffers extension
// in the shader, broadcast is emualted by replacing gl_FragColor with gl_FragData.
TEST_F(EmulateGLFragColorBroadcastTest, FragColorBroadcast)
{
    const std::string shaderString =
        "#extension GL_EXT_draw_buffers : require\n"
        "void main()\n"
        "{\n"
        "    gl_FragColor = vec4(1, 0, 0, 0);\n"
        "}\n";
    compile(shaderString);
    EXPECT_FALSE(foundInCode("gl_FragColor"));
    EXPECT_TRUE(foundInCode("gl_FragData[0]"));
    EXPECT_TRUE(foundInCode("gl_FragData[1]"));
}

// Verifies that with explicitly enabling GL_EXT_draw_buffers extension
// in the shader with an empty main(), anothing happens.
TEST_F(EmulateGLFragColorBroadcastTest, EmptyMain)
{
    const std::string shaderString =
        "#extension GL_EXT_draw_buffers : require\n"
        "void main()\n"
        "{\n"
        "}\n";
    compile(shaderString);
    EXPECT_FALSE(foundInCode("gl_FragColor"));
    EXPECT_FALSE(foundInCode("gl_FragData[0]"));
    EXPECT_FALSE(foundInCode("gl_FragData[1]"));
}

}  // namespace
