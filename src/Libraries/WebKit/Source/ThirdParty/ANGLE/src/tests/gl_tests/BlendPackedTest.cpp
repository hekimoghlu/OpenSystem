/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 20, 2022.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

using namespace angle;

class BlendPackedTest : public ANGLETest<>
{
  protected:
    BlendPackedTest()
    {
        setWindowWidth(128);
        setWindowHeight(128);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
    }

    template <GLenum internalformat, GLuint components>
    void runTest()
    {
        constexpr char kFs[] =
            "#version 100\n"
            "void main(void)\n"
            "{\n"
            "    gl_FragColor = vec4(0.0, 0.0, 1.0, 1.0);\n"
            "}\n";

        ANGLE_GL_PROGRAM(program, essl1_shaders::vs::Simple(), kFs);
        glUseProgram(program);

        GLFramebuffer framebuffer;
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        GLRenderbuffer colorRenderbuffer;
        glBindRenderbuffer(GL_RENDERBUFFER, colorRenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, internalformat, getWindowWidth(), getWindowHeight());
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                                  colorRenderbuffer);

        glClearColor(1.0, 1.0, 0.0, 0.0);
        glClear(GL_COLOR_BUFFER_BIT);
        ASSERT_GL_NO_ERROR();

        if (components == 3)
        {
            EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::yellow);
        }
        else
        {
            EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor(255u, 255u, 0, 0));
        }

        glEnable(GL_BLEND);
        glBlendEquation(GL_FUNC_ADD);
        glBlendFunc(GL_ONE, GL_ONE);

        drawQuad(program, essl1_shaders::PositionAttrib(), 0.5f);
        ASSERT_GL_NO_ERROR();

        EXPECT_PIXEL_COLOR_EQ(0, 0, GLColor::white);
    }
};

// Test that blending is applied to attachments with packed formats.
TEST_P(BlendPackedTest, RGB565)
{
    runTest<GL_RGB565, 3>();
}

TEST_P(BlendPackedTest, RGBA4)
{
    runTest<GL_RGBA4, 4>();
}

TEST_P(BlendPackedTest, RGB5_A1)
{
    runTest<GL_RGB5_A1, 4>();
}

TEST_P(BlendPackedTest, RGB10_A2)
{
    ANGLE_SKIP_TEST_IF(getClientMajorVersion() < 3);
    runTest<GL_RGB10_A2, 4>();
}

// Use this to select which configurations (e.g. which renderer, which GLES major version) these
// tests should be run against.
ANGLE_INSTANTIATE_TEST_ES2_AND_ES3(BlendPackedTest);
