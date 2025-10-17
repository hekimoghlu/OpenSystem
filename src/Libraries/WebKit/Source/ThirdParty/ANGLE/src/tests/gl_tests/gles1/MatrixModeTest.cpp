/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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

// MatrixModeTest.cpp: Tests basic usage of glMatrixMode.

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include <vector>

using namespace angle;

class MatrixModeTest : public ANGLETest<>
{
  protected:
    MatrixModeTest()
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
TEST_P(MatrixModeTest, InitialState)
{
    GLint matrixMode;
    glGetIntegerv(GL_MATRIX_MODE, &matrixMode);
    EXPECT_GL_NO_ERROR();
    EXPECT_GLENUM_EQ(GL_MODELVIEW, matrixMode);
}

// Checks for error-generating cases.
TEST_P(MatrixModeTest, Negative)
{
    glMatrixMode(0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
    glMatrixMode(GL_TEXTURE_2D);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
}

// Checks that matrix mode can be set.
TEST_P(MatrixModeTest, Set)
{
    GLint matrixMode;

    std::vector<GLenum> modes = {GL_PROJECTION, GL_MODELVIEW, GL_TEXTURE};

    for (auto mode : modes)
    {
        glMatrixMode(mode);
        EXPECT_GL_NO_ERROR();
        glGetIntegerv(GL_MATRIX_MODE, &matrixMode);
        EXPECT_GLENUM_EQ(mode, matrixMode);
    }
}

ANGLE_INSTANTIATE_TEST_ES1(MatrixModeTest);
