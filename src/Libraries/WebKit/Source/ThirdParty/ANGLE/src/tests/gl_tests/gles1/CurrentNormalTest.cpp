/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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

// CurrentNormalTest.cpp: Tests basic usage of glNormal3(f|x).

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include "util/random_utils.h"

#include "common/vector_utils.h"

#include <stdint.h>

using namespace angle;

class CurrentNormalTest : public ANGLETest<>
{
  protected:
    CurrentNormalTest()
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
TEST_P(CurrentNormalTest, InitialState)
{
    const angle::Vector3 kUp(0.0f, 0.0f, 1.0f);
    angle::Vector3 actualNormal;
    glGetFloatv(GL_CURRENT_NORMAL, actualNormal.data());
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(kUp, actualNormal);
}

// Set test: Checks that the current normal is properly set and retrieved.
TEST_P(CurrentNormalTest, Set)
{
    glNormal3f(0.1f, 0.2f, 0.3f);
    EXPECT_GL_NO_ERROR();

    angle::Vector3 actualNormal;

    glGetFloatv(GL_CURRENT_NORMAL, actualNormal.data());
    EXPECT_GL_NO_ERROR();
    EXPECT_EQ(angle::Vector3(0.1f, 0.2f, 0.3f), actualNormal);

    float epsilon = 0.00001f;

    glNormal3x(0x10000, 0x3333, 0x5555);
    EXPECT_GL_NO_ERROR();

    glGetFloatv(GL_CURRENT_NORMAL, actualNormal.data());
    EXPECT_GL_NO_ERROR();
    EXPECT_NEAR(1.0f, actualNormal[0], epsilon);
    EXPECT_NEAR(0.2f, actualNormal[1], epsilon);
    EXPECT_NEAR(1.0f / 3.0f, actualNormal[2], epsilon);
}

ANGLE_INSTANTIATE_TEST_ES1(CurrentNormalTest);
