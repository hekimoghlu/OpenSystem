/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

// VertexPointerTest.cpp: Tests basic usage of built-in vertex attributes of GLES1.

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

using namespace angle;

class VertexPointerTest : public ANGLETest<>
{
  protected:
    VertexPointerTest()
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

// Checks that we can assign to client side vertex arrays
TEST_P(VertexPointerTest, AssignRetrieve)
{
    std::vector<float> testVertexAttribute = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };

    glVertexPointer(4, GL_FLOAT, 0, testVertexAttribute.data());
    EXPECT_GL_NO_ERROR();

    void *ptr = nullptr;
    glGetPointerv(GL_VERTEX_ARRAY_POINTER, &ptr);
    EXPECT_EQ(testVertexAttribute.data(), ptr);

    glColorPointer(4, GL_FLOAT, 0, testVertexAttribute.data() + 4);
    glGetPointerv(GL_COLOR_ARRAY_POINTER, &ptr);
    EXPECT_EQ(testVertexAttribute.data() + 4, ptr);

    glNormalPointer(GL_FLOAT, 0, testVertexAttribute.data() + 8);
    glGetPointerv(GL_NORMAL_ARRAY_POINTER, &ptr);
    EXPECT_EQ(testVertexAttribute.data() + 8, ptr);

    glPointSizePointerOES(GL_FLOAT, 0, testVertexAttribute.data() + 8);
    glGetPointerv(GL_POINT_SIZE_ARRAY_POINTER_OES, &ptr);
    EXPECT_EQ(testVertexAttribute.data() + 8, ptr);

    GLint maxTextureUnits;
    glGetIntegerv(GL_MAX_TEXTURE_UNITS, &maxTextureUnits);
    for (int i = 0; i < maxTextureUnits; i++)
    {
        glClientActiveTexture(GL_TEXTURE0 + i);
        glTexCoordPointer(4, GL_FLOAT, 0, testVertexAttribute.data() + i * 4);
        glGetPointerv(GL_TEXTURE_COORD_ARRAY_POINTER, &ptr);
        EXPECT_EQ(testVertexAttribute.data() + i * 4, ptr);
    }
}

// Checks that we can assign to client side vertex arrays with color vertex attributes of type
// GLubyte
TEST_P(VertexPointerTest, AssignRetrieveColorUnsignedByte)
{
    std::vector<float> testVertexAttribute = {
        1.0f,
        1.0f,
        1.0f,
        1.0f,
    };

    std::vector<GLubyte> testColorAttribute = {
        1,
        1,
        1,
        1,
    };

    glVertexPointer(4, GL_FLOAT, 0, testVertexAttribute.data());
    EXPECT_GL_NO_ERROR();

    void *ptr = nullptr;
    glGetPointerv(GL_VERTEX_ARRAY_POINTER, &ptr);
    EXPECT_EQ(testVertexAttribute.data(), ptr);

    glColorPointer(4, GL_UNSIGNED_BYTE, 0, testColorAttribute.data());
    glGetPointerv(GL_COLOR_ARRAY_POINTER, &ptr);
    EXPECT_EQ(testColorAttribute.data(), ptr);
    ASSERT_GL_NO_ERROR();
}

ANGLE_INSTANTIATE_TEST_ES1(VertexPointerTest);
