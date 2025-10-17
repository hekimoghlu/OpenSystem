/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// ObjectAllocationTest
//   Tests for object allocations and lifetimes.
//

#include "test_utils/ANGLETest.h"

using namespace angle;

namespace
{

class ObjectAllocationTest : public ANGLETest<>
{
  protected:
    ObjectAllocationTest() {}
};

class ObjectAllocationTestES3 : public ObjectAllocationTest
{};

// Test that we don't re-allocate a bound framebuffer ID.
TEST_P(ObjectAllocationTestES3, BindFramebufferBeforeGen)
{
    glBindFramebuffer(GL_FRAMEBUFFER, 1);
    GLuint fbo = 0;
    glGenFramebuffers(1, &fbo);
    EXPECT_NE(1u, fbo);
    glDeleteFramebuffers(1, &fbo);
    EXPECT_GL_NO_ERROR();
}

// Test that we don't re-allocate a bound framebuffer ID, other pattern.
TEST_P(ObjectAllocationTestES3, BindFramebufferAfterGen)
{
    GLuint reservedFBO1 = 1;
    GLuint reservedFBO2 = 2;

    GLuint firstFBO = 0;
    glGenFramebuffers(1, &firstFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, reservedFBO1);
    glDeleteFramebuffers(1, &firstFBO);

    glBindFramebuffer(GL_FRAMEBUFFER, reservedFBO2);
    GLuint secondFBOs[2] = {0};
    glGenFramebuffers(2, secondFBOs);
    EXPECT_NE(reservedFBO2, secondFBOs[0]);
    EXPECT_NE(reservedFBO2, secondFBOs[1]);
    glDeleteFramebuffers(2, secondFBOs);

    // Clean up
    glDeleteFramebuffers(1, &reservedFBO1);
    glDeleteFramebuffers(1, &reservedFBO2);

    EXPECT_GL_NO_ERROR();
}

// Test that we don't re-allocate a bound framebuffer ID.
TEST_P(ObjectAllocationTest, BindRenderbuffer)
{
    GLuint rbId;
    glGenRenderbuffers(1, &rbId);
    glBindRenderbuffer(GL_RENDERBUFFER, rbId);
    EXPECT_GL_NO_ERROR();

    // Swap now to trigger the serialization of the renderbuffer that
    // was initialized with the default values
    swapBuffers();

    glDeleteRenderbuffers(1, &rbId);
    EXPECT_GL_NO_ERROR();
}

// Renderbuffers can be created on the fly by calling glBindRenderbuffer,
// so// check that the call doesn't fail that the renderbuffer is also deleted
TEST_P(ObjectAllocationTest, BindRenderbufferBeforeGenAndDelete)
{
    GLuint rbId = 1;
    glBindRenderbuffer(GL_RENDERBUFFER, rbId);
    EXPECT_GL_NO_ERROR();

    // Swap now to trigger the serialization of the renderbuffer that
    // was initialized with the default values
    swapBuffers();

    glDeleteRenderbuffers(1, &rbId);
    EXPECT_GL_NO_ERROR();
}

// Buffers can be created on the fly by calling glBindBuffer, so
// check that the call doesn't fail that the buffer is also deleted
TEST_P(ObjectAllocationTest, BindBufferBeforeGenAndDelete)
{
    GLuint id = 1;
    glBindBuffer(GL_ARRAY_BUFFER, id);
    EXPECT_GL_NO_ERROR();
    // trigger serialization to capture the created buffer ID
    swapBuffers();
    glDeleteBuffers(1, &id);
    EXPECT_GL_NO_ERROR();
}

// Textures can be created on the fly by calling glBindTexture, so
// check that the call doesn't fail that the texture is also deleted
TEST_P(ObjectAllocationTest, BindTextureBeforeGenAndDelete)
{
    GLuint id = 1;
    glBindTexture(GL_TEXTURE_2D, id);
    EXPECT_GL_NO_ERROR();
    // trigger serialization to capture the created texture ID
    swapBuffers();
    glDeleteTextures(1, &id);
    EXPECT_GL_NO_ERROR();
}

}  // anonymous namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ObjectAllocationTest);
ANGLE_INSTANTIATE_TEST_ES2(ObjectAllocationTest);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ObjectAllocationTestES3);
ANGLE_INSTANTIATE_TEST_ES3(ObjectAllocationTestES3);
