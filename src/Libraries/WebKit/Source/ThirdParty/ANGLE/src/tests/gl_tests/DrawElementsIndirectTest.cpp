/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DrawElementsIndirect tests:
// Test issuing DrawElementsIndirect commands.
//

#include "test_utils/ANGLETest.h"
#include "test_utils/gl_raii.h"

using namespace angle;

namespace
{

class DrawElementsIndirectTest : public ANGLETest<>
{
  protected:
    DrawElementsIndirectTest()
    {
        setWindowWidth(64);
        setWindowHeight(64);
        setConfigRedBits(8);
        setConfigGreenBits(8);
        setConfigBlueBits(8);
        setConfigAlphaBits(8);
    }
};

// Test that DrawElementsIndirect works when count is zero.
TEST_P(DrawElementsIndirectTest, CountIsZero)
{
    constexpr char kVS[] =
        "attribute vec3 a_pos;\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(a_pos, 1.0);\n"
        "}\n";

    GLProgram program;
    program.makeRaster(kVS, essl1_shaders::fs::Blue());
    glUseProgram(program);

    GLVertexArray va;
    glBindVertexArray(va);

    GLBuffer vertexBuffer;
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
    static float vertexData[3] = {0};
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);

    GLBuffer indexBuffer;
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
    static GLuint indexData[3] = {0};
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indexData), indexData, GL_STATIC_DRAW);

    GLBuffer indirectBuffer;
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indirectBuffer);
    static GLuint indirectData[5] = {0};
    glBufferData(GL_DRAW_INDIRECT_BUFFER, sizeof(indirectData), indirectData, GL_STATIC_DRAW);

    GLint posLocation = glGetAttribLocation(program, "a_pos");
    glEnableVertexAttribArray(posLocation);
    glVertexAttribPointer(posLocation, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    glDrawElementsIndirect(GL_TRIANGLE_FAN, GL_UNSIGNED_INT, nullptr);

    ASSERT_GL_NO_ERROR();
}

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(DrawElementsIndirectTest);
ANGLE_INSTANTIATE_TEST_ES31(DrawElementsIndirectTest);
}  // namespace