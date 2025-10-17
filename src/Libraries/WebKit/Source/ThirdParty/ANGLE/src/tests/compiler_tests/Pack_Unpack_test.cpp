/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// Pack_Unpack_test.cpp:
//   Tests for the emulating pack_unpack functions for GLSL.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

namespace
{

class PackUnpackTest : public MatchOutputCodeTest
{
  public:
    PackUnpackTest() : MatchOutputCodeTest(GL_FRAGMENT_SHADER, SH_GLSL_400_CORE_OUTPUT) {}
};

// Check if PackSnorm2x16 Emulation for GLSL < 4.2 compile correctly.
TEST_F(PackUnpackTest, PackSnorm2x16Emulation)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision mediump float;
        layout(location = 0) out mediump vec4 fragColor;
        void main()
        {
           vec2 v;
           uint u = packSnorm2x16(v);
           fragColor = vec4(u);
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("uint packSnorm2x16_emu(vec2 v)"));
}

// Check if UnpackSnorm2x16 Emulation for GLSL < 4.2 compile correctly.
TEST_F(PackUnpackTest, UnpackSnorm2x16Emulation)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision mediump float;
        layout(location = 0) out mediump vec4 fragColor;
        void main()
        {
           uint u;
           vec2 v = unpackSnorm2x16(u);
           fragColor = vec4(v, 0.0, 0.0);
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("vec2 unpackSnorm2x16_emu(uint u)"));
}

// Check if PackUnorm2x16 Emulation for GLSL < 4.1 compiles correctly.
TEST_F(PackUnpackTest, PackUnorm2x16Emulation)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision mediump float;
        layout(location = 0) out mediump vec4 fragColor;
        void main()
        {
           vec2 v;
           uint u = packUnorm2x16(v);
           fragColor = vec4(u);
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("uint packUnorm2x16_emu(vec2 v)"));
}

// Check if UnpackSnorm2x16 Emulation for GLSL < 4.1 compiles correctly.
TEST_F(PackUnpackTest, UnpackUnorm2x16Emulation)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision mediump float;
        layout(location = 0) out mediump vec4 fragColor;
        void main()
        {
           uint u;
           vec2 v = unpackUnorm2x16(u);
           fragColor = vec4(v, 0.0, 0.0);
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("vec2 unpackUnorm2x16_emu(uint u)"));
}

// Check if PackHalf2x16 Emulation for GLSL < 4.2 compiles correctly.
TEST_F(PackUnpackTest, PackHalf2x16Emulation)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision mediump float;
        layout(location = 0) out mediump vec4 fragColor;
        void main()
        {
            vec2 v;
            uint u = packHalf2x16(v);
            fragColor = vec4(u);
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("uint packHalf2x16_emu(vec2 v)"));
}

// Check if UnpackHalf2x16 Emulation for GLSL < 4.2 compiles correctly.
TEST_F(PackUnpackTest, UnpackHalf2x16Emulation)
{
    const std::string &shaderString =
        R"(#version 300 es
        precision mediump float;
        layout(location = 0) out mediump vec4 fragColor;
        void main()
        {
            uint u;
            vec2 v = unpackHalf2x16(u);
            fragColor = vec4(v, 0.0, 0.0);
        })";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("vec2 unpackHalf2x16_emu(uint u)"));
}

}  // namespace
