/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// ShadingRateQcomTest.cpp : Tests of the GL_QCOM_shading_rate extension.

#include "test_utils/ANGLETest.h"

#include "test_utils/gl_raii.h"

namespace angle
{

class ShadingRateQcomTest : public ANGLETest<>
{
  protected:
    ShadingRateQcomTest()
    {
        setWindowWidth(256);
        setWindowHeight(256);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
    }
};

// Test basic functionality of QCOM_shading_rate
TEST_P(ShadingRateQcomTest, Basic)
{
    ANGLE_SKIP_TEST_IF(!IsGLExtensionEnabled("GL_QCOM_shading_rate"));

    glClearColor(0.25f, 0.5f, 0.5f, 0.5f);
    glShadingRateQCOM(GL_SHADING_RATE_1X1_PIXELS_QCOM);
    glClear(GL_COLOR_BUFFER_BIT);
    EXPECT_PIXEL_NEAR(0, 0, 64, 128, 128, 128, 1.0);

    ANGLE_GL_PROGRAM(uniformColorProgram, essl1_shaders::vs::Simple(),
                     essl1_shaders::fs::UniformColor());
    glUseProgram(uniformColorProgram);
    GLint colorUniformLocation =
        glGetUniformLocation(uniformColorProgram, angle::essl1_shaders::ColorUniform());
    ASSERT_NE(colorUniformLocation, -1);

    glShadingRateQCOM(GL_SHADING_RATE_1X2_PIXELS_QCOM);
    glUniform4f(colorUniformLocation, 1.0f, 0.0f, 0.0f, 1.0f);
    drawQuad(uniformColorProgram, essl1_shaders::PositionAttrib(), 0.5f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::red);

    glShadingRateQCOM(GL_SHADING_RATE_2X1_PIXELS_QCOM);
    glUniform4f(colorUniformLocation, 0.0f, 1.0f, 0.0f, 1.0f);
    drawQuad(uniformColorProgram, essl1_shaders::PositionAttrib(), 0.5f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::green);

    glShadingRateQCOM(GL_SHADING_RATE_2X2_PIXELS_QCOM);
    glUniform4f(colorUniformLocation, 0.0f, 0.0f, 1.0f, 1.0f);
    drawQuad(uniformColorProgram, essl1_shaders::PositionAttrib(), 0.5f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::blue);

    glShadingRateQCOM(GL_SHADING_RATE_4X2_PIXELS_QCOM);
    glUniform4f(colorUniformLocation, 1.0f, 1.0f, 0.0f, 1.0f);
    drawQuad(uniformColorProgram, essl1_shaders::PositionAttrib(), 0.5f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::yellow);

    glShadingRateQCOM(GL_SHADING_RATE_4X4_PIXELS_QCOM);
    glUniform4f(colorUniformLocation, 0.0f, 1.0f, 1.0f, 1.0f);
    drawQuad(uniformColorProgram, essl1_shaders::PositionAttrib(), 0.5f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::cyan);

    glEnable(GL_SHADING_RATE_PRESERVE_ASPECT_RATIO_QCOM);
    glUniform4f(colorUniformLocation, 1.0f, 0.0f, 1.0f, 1.0f);
    drawQuad(uniformColorProgram, essl1_shaders::PositionAttrib(), 0.5f);
    ASSERT_GL_NO_ERROR();
    EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::magenta);
}

// Use this to select which configurations (e.g. which renderer, which GLES major version) these
// tests should be run against.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ShadingRateQcomTest);
ANGLE_INSTANTIATE_TEST_ES2_AND_ES3_AND_ES31(ShadingRateQcomTest);

}  // namespace angle
