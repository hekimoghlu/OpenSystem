/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 28, 2021.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// FloatingPointSurfaceTest.cpp : Test functionality of the EGL_EXT_pixel_format_float extension.

#include "test_utils/ANGLETest.h"

using namespace angle;

class FloatingPointSurfaceTest : public ANGLETest<>
{
  protected:
    FloatingPointSurfaceTest()
    {
        setWindowWidth(512);
        setWindowHeight(512);
        setConfigRedBits(16);
        setConfigGreenBits(16);
        setConfigBlueBits(16);
        setConfigAlphaBits(16);
        setConfigComponentType(EGL_COLOR_COMPONENT_TYPE_FLOAT_EXT);
    }

    void testSetUp() override
    {
        constexpr char kFS[] =
            "precision highp float;\n"
            "void main()\n"
            "{\n"
            "   gl_FragColor = vec4(1.0, 2.0, 3.0, 4.0);\n"
            "}\n";

        mProgram = CompileProgram(essl1_shaders::vs::Simple(), kFS);
        ASSERT_NE(0u, mProgram) << "shader compilation failed.";

        ASSERT_GL_NO_ERROR();
    }

    void testTearDown() override { glDeleteProgram(mProgram); }

    GLuint mProgram;
};

// Test clearing and checking the color is correct
TEST_P(FloatingPointSurfaceTest, Clearing)
{
    GLColor32F clearColor(0.0f, 1.0f, 2.0f, 3.0f);
    glClearColor(clearColor.R, clearColor.G, clearColor.B, clearColor.A);
    glClear(GL_COLOR_BUFFER_BIT);
    ASSERT_GL_NO_ERROR();

    EXPECT_PIXEL_COLOR32F_EQ(0, 0, clearColor);
}

// Test drawing and checking the color is correct
TEST_P(FloatingPointSurfaceTest, Drawing)
{
    glUseProgram(mProgram);
    drawQuad(mProgram, essl1_shaders::PositionAttrib(), 0.5f);

    EXPECT_PIXEL_32F_EQ(0, 0, 1.0f, 2.0f, 3.0f, 4.0f);
}

// Use this to select which configurations (e.g. which renderer, which GLES major version) these
// tests should be run against.
ANGLE_INSTANTIATE_TEST(FloatingPointSurfaceTest,
                       ES2_D3D11(),
                       ES3_D3D11(),
                       ES2_D3D11_PRESENT_PATH_FAST());

// This test suite is not instantiated on some OSes.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(FloatingPointSurfaceTest);
