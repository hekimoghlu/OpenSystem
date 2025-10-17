/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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

// CurrentColorTest.cpp: Tests basic usage of glColor4(f|ub|x).

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include "util/random_utils.h"

#include <stdint.h>

using namespace angle;

class CurrentColorTest : public ANGLETest<>
{
  protected:
    CurrentColorTest()
    {
        setWindowWidth(32);
        setWindowHeight(32);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
        setConfigDepthBits(24);
    }
};

// State query: Checks the initial state is correct.
TEST_P(CurrentColorTest, InitialState)
{
    const GLColor32F kFloatWhite(1.0f, 1.0f, 1.0f, 1.0f);
    GLColor32F actualColor;
    glGetFloatv(GL_CURRENT_COLOR, &actualColor.R);
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(kFloatWhite, actualColor);
}

// Set test: Checks that the current color is properly set and retrieved.
TEST_P(CurrentColorTest, Set)
{
    float epsilon = 0.00001f;

    glColor4f(0.1f, 0.2f, 0.3f, 0.4f);
    EXPECT_GL_NO_ERROR();

    GLColor32F floatColor;
    glGetFloatv(GL_CURRENT_COLOR, &floatColor.R);
    EXPECT_GL_NO_ERROR();

    EXPECT_EQ(GLColor32F(0.1f, 0.2f, 0.3f, 0.4f), floatColor);

    glColor4ub(0xff, 0x0, 0x55, 0x33);

    glGetFloatv(GL_CURRENT_COLOR, &floatColor.R);
    EXPECT_GL_NO_ERROR();

    EXPECT_NEAR(1.0f, floatColor.R, epsilon);
    EXPECT_NEAR(0.0f, floatColor.G, epsilon);
    EXPECT_NEAR(1.0f / 3.0f, floatColor.B, epsilon);
    EXPECT_NEAR(0.2f, floatColor.A, epsilon);

    glColor4x(0x10000, 0x0, 0x3333, 0x5555);

    glGetFloatv(GL_CURRENT_COLOR, &floatColor.R);
    EXPECT_GL_NO_ERROR();

    EXPECT_NEAR(1.0f, floatColor.R, epsilon);
    EXPECT_NEAR(0.0f, floatColor.G, epsilon);
    EXPECT_NEAR(0.2f, floatColor.B, epsilon);
    EXPECT_NEAR(1.0f / 3.0f, floatColor.A, epsilon);
}

ANGLE_INSTANTIATE_TEST_ES1(CurrentColorTest);
