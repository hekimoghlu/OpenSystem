/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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

// ClientStateEnable.cpp: Tests basic usage of gl(Enable|Disable)ClientState.

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

#include <vector>

using namespace angle;

class ClientStateEnable : public ANGLETest<>
{
  protected:
    ClientStateEnable()
    {
        setWindowWidth(32);
        setWindowHeight(32);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
        setConfigDepthBits(24);
    }

    std::vector<GLenum> mClientStates = {
        GL_VERTEX_ARRAY,         GL_NORMAL_ARRAY,        GL_COLOR_ARRAY,
        GL_POINT_SIZE_ARRAY_OES, GL_TEXTURE_COORD_ARRAY,
    };
};

// Checks that all client vertex array states are disabled to start with.
TEST_P(ClientStateEnable, InitialState)
{
    for (auto clientState : mClientStates)
    {
        EXPECT_GL_FALSE(glIsEnabled(clientState));
        EXPECT_GL_NO_ERROR();
    }
}

// Checks that glEnableClientState sets the state to be enabled,
// and glDisableClientState sets the state to be disabled.
TEST_P(ClientStateEnable, EnableState)
{
    for (auto clientState : mClientStates)
    {
        EXPECT_GL_FALSE(glIsEnabled(clientState));
        glEnableClientState(clientState);
        EXPECT_GL_NO_ERROR();
        EXPECT_GL_TRUE(glIsEnabled(clientState));
        glDisableClientState(clientState);
        EXPECT_GL_NO_ERROR();
        EXPECT_GL_FALSE(glIsEnabled(clientState));
    }
}

// Negative test: Checks that invalid enums for client state generate the proper GL error.
TEST_P(ClientStateEnable, Negative)
{
    glEnableClientState(0);
    EXPECT_GL_ERROR(GL_INVALID_ENUM);
}

// Checks that enable/disable states are different if we are in different client texture unit
// states.
TEST_P(ClientStateEnable, TextureUnit)
{
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    // Spec minimum lets us assume 2 multitexturing units.
    glClientActiveTexture(GL_TEXTURE1);
    EXPECT_GL_FALSE(glIsEnabled(GL_TEXTURE_COORD_ARRAY));

    glEnableClientState(GL_TEXTURE_COORD_ARRAY);
    EXPECT_GL_TRUE(glIsEnabled(GL_TEXTURE_COORD_ARRAY));

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    EXPECT_GL_FALSE(glIsEnabled(GL_TEXTURE_COORD_ARRAY));

    glClientActiveTexture(GL_TEXTURE0);
    EXPECT_GL_TRUE(glIsEnabled(GL_TEXTURE_COORD_ARRAY));

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    EXPECT_GL_FALSE(glIsEnabled(GL_TEXTURE_COORD_ARRAY));
}

ANGLE_INSTANTIATE_TEST_ES1(ClientStateEnable);
