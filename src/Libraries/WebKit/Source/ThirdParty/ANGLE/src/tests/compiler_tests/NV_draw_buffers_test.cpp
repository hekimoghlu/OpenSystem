/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// NV_draw_buffers_test.cpp:
//   Test for NV_draw_buffers setting
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

class NVDrawBuffersTest : public MatchOutputCodeTest
{
  public:
    NVDrawBuffersTest() : MatchOutputCodeTest(GL_FRAGMENT_SHADER, SH_ESSL_OUTPUT)
    {
        ShBuiltInResources *resources = getResources();
        resources->MaxDrawBuffers     = 8;
        resources->EXT_draw_buffers   = 1;
        resources->NV_draw_buffers    = 1;
    }
};

TEST_F(NVDrawBuffersTest, NVDrawBuffers)
{
    const std::string &shaderString =
        "#extension GL_EXT_draw_buffers : require\n"
        "precision mediump float;\n"
        "void main() {\n"
        "   gl_FragData[0] = vec4(1.0);\n"
        "   gl_FragData[1] = vec4(0.0);\n"
        "}\n";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("GL_NV_draw_buffers"));
    ASSERT_FALSE(foundInCode("GL_EXT_draw_buffers"));
}
