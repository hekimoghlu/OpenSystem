/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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

// TextureTargetEnableTest.cpp: Tests basic usage of built-in vertex attributes of GLES1.

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

using namespace angle;

class TextureTargetEnableTest : public ANGLETest<>
{
  protected:
    TextureTargetEnableTest()
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

// Checks that 2D/Cube targets are disabled initially.
TEST_P(TextureTargetEnableTest, InitialState)
{
    EXPECT_GL_FALSE(glIsEnabled(GL_TEXTURE_2D));
    EXPECT_GL_NO_ERROR();
    EXPECT_GL_FALSE(glIsEnabled(GL_TEXTURE_CUBE_MAP));
    EXPECT_GL_NO_ERROR();
}

// Checks that 2D/cube targets can be set to enabled or disabled.
TEST_P(TextureTargetEnableTest, Set)
{
    glEnable(GL_TEXTURE_2D);
    EXPECT_GL_NO_ERROR();
    EXPECT_GL_TRUE(glIsEnabled(GL_TEXTURE_2D));

    glEnable(GL_TEXTURE_CUBE_MAP);
    EXPECT_GL_NO_ERROR();
    EXPECT_GL_TRUE(glIsEnabled(GL_TEXTURE_CUBE_MAP));
}

// Checks that targets can be set to enabled or disabled and it does not affect the setting of
// other texture units.
TEST_P(TextureTargetEnableTest, SetSeparateUnits)
{
    GLint units;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &units);

    for (int i = 0; i < units; i++)
    {

        glActiveTexture(GL_TEXTURE0 + i);
        EXPECT_GL_NO_ERROR();

        EXPECT_GL_FALSE(glIsEnabled(GL_TEXTURE_2D));
        EXPECT_GL_NO_ERROR();

        EXPECT_GL_FALSE(glIsEnabled(GL_TEXTURE_CUBE_MAP));
        EXPECT_GL_NO_ERROR();

        glEnable(GL_TEXTURE_2D);
        EXPECT_GL_NO_ERROR();
        EXPECT_GL_TRUE(glIsEnabled(GL_TEXTURE_2D));

        glEnable(GL_TEXTURE_CUBE_MAP);
        EXPECT_GL_NO_ERROR();
        EXPECT_GL_TRUE(glIsEnabled(GL_TEXTURE_CUBE_MAP));
    }
}

ANGLE_INSTANTIATE_TEST_ES1(TextureTargetEnableTest);
