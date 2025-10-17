/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 27, 2024.
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

// ColorMaterialTest.cpp: Tests basic usage of texture environments.

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include "util/random_utils.h"

#include <stdint.h>

using namespace angle;

class ColorMaterialTest : public ANGLETest<>
{
  protected:
    ColorMaterialTest()
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

// Ensure that when GL_COLOR_MATERIAL is enabled, material properties are
// inherited from glColor4f
TEST_P(ColorMaterialTest, ColorMaterialOn)
{
    std::array<GLfloat, 4> expected = {0.1f, 0.2f, 0.3f, 0.4f};

    glColor4f(0.1f, 0.2f, 0.3f, 0.4f);

    std::array<GLfloat, 4> ambientAndDiffuse = {0.3f, 0.2f, 0.1f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, ambientAndDiffuse.data());

    glEnable(GL_COLOR_MATERIAL);

    std::array<GLfloat, 4> ambient = {};
    glGetMaterialfv(GL_FRONT, GL_AMBIENT, ambient.data());

    std::array<GLfloat, 4> diffuse = {};
    glGetMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse.data());

    EXPECT_EQ(ambient, expected);
    EXPECT_EQ(diffuse, expected);
}

// Ensure that when GL_COLOR_MATERIAL is disabled, material properties are
// inherited from glMaterialfv
TEST_P(ColorMaterialTest, ColorMaterialOff)
{
    std::array<GLfloat, 4> expectedAmbientAndDiffuse = {0.3f, 0.2f, 0.1f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, expectedAmbientAndDiffuse.data());

    glColor4f(0.1f, 0.2f, 0.3f, 0.4f);

    std::array<GLfloat, 4> ambient = {};
    glGetMaterialfv(GL_FRONT, GL_AMBIENT, ambient.data());

    std::array<GLfloat, 4> diffuse = {};
    glGetMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse.data());

    EXPECT_EQ(ambient, expectedAmbientAndDiffuse);
}

ANGLE_INSTANTIATE_TEST_ES1(ColorMaterialTest);
