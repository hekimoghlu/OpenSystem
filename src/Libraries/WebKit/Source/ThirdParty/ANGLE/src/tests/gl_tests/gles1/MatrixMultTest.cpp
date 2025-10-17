/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 10, 2022.
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

// MatrixMultTest.cpp: Tests basic usage of glMultMatrix(f|x).

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include "common/matrix_utils.h"
#include "util/random_utils.h"

#include <stdint.h>

using namespace angle;

class MatrixMultTest : public ANGLETest<>
{
  protected:
    MatrixMultTest()
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

// Multiplies identity matrix with itself.
TEST_P(MatrixMultTest, Identity)
{
    angle::Mat4 testMatrix;
    angle::Mat4 outputMatrix;

    glMultMatrixf(testMatrix.data());
    EXPECT_GL_NO_ERROR();

    glGetFloatv(GL_MODELVIEW_MATRIX, outputMatrix.data());
    EXPECT_GL_NO_ERROR();

    EXPECT_EQ(angle::Mat4(), outputMatrix);
}

// Multiplies translation matrix and checks matrix underneath in stack is not affected.
TEST_P(MatrixMultTest, Translation)
{
    glPushMatrix();

    angle::Mat4 testMatrix = angle::Mat4::Translate(angle::Vector3(1.0f, 0.0f, 0.0f));

    angle::Mat4 outputMatrix;

    glMultMatrixf(testMatrix.data());
    EXPECT_GL_NO_ERROR();

    glMultMatrixf(testMatrix.data());
    EXPECT_GL_NO_ERROR();

    glGetFloatv(GL_MODELVIEW_MATRIX, outputMatrix.data());
    EXPECT_GL_NO_ERROR();

    EXPECT_EQ(angle::Mat4::Translate(angle::Vector3(2.0f, 0.0f, 0.0f)), outputMatrix);

    glPopMatrix();

    glGetFloatv(GL_MODELVIEW_MATRIX, outputMatrix.data());
    EXPECT_GL_NO_ERROR();

    EXPECT_EQ(angle::Mat4(), outputMatrix);
}

ANGLE_INSTANTIATE_TEST_ES1(MatrixMultTest);
