/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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

// ClientActiveTextureTest.cpp: Tests basic usage of glClientActiveTexture.

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include "util/random_utils.h"

#include <stdint.h>

using namespace angle;

class ClientActiveTextureTest : public ANGLETest<>
{
  protected:
    ClientActiveTextureTest()
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
TEST_P(ClientActiveTextureTest, InitialState)
{
    GLint clientActiveTexture = 0;
    glGetIntegerv(GL_CLIENT_ACTIVE_TEXTURE, &clientActiveTexture);
    EXPECT_GL_NO_ERROR();
    EXPECT_GLENUM_EQ(GL_TEXTURE0, clientActiveTexture);
}

// Negative test: Checks against invalid use of glClientActiveTexture.
TEST_P(ClientActiveTextureTest, Negative)
{
    glClientActiveTexture(0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);

    GLint maxTextureUnits = 0;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);

    glClientActiveTexture(GL_TEXTURE0 + maxTextureUnits);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
}

// Checks that the number of multitexturing units is above spec minimum.
TEST_P(ClientActiveTextureTest, Limits)
{
    GLint maxTextureUnits = 0;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);
    EXPECT_GE(maxTextureUnits, 2);
}

// Set test: Checks that GL_TEXTURE0..GL_TEXTURE[GL_MAX_TEXTURE_UNITS-1] can be set.
TEST_P(ClientActiveTextureTest, Set)
{
    GLint maxTextureUnits = 0;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);

    for (GLint i = 0; i < maxTextureUnits; i++)
    {
        glClientActiveTexture(GL_TEXTURE0 + i);
        EXPECT_GL_NO_ERROR();
        GLint clientActiveTexture = 0;
        glGetIntegerv(GL_CLIENT_ACTIVE_TEXTURE, (GLint *)&clientActiveTexture);
        EXPECT_GL_NO_ERROR();
        EXPECT_GLENUM_EQ(GL_TEXTURE0 + i, clientActiveTexture);
    }
}

ANGLE_INSTANTIATE_TEST_ES1(ClientActiveTextureTest);
