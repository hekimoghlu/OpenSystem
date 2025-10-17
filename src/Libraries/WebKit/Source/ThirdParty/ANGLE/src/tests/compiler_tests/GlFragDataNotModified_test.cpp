/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 2, 2023.
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
// GlFragDataNotModified_test.cpp:
//   Test that the properties of built-in gl_FragData are not modified when a shader is compiled
//   multiple times.
//

#include "tests/test_utils/ShaderCompileTreeTest.h"

namespace
{

class GlFragDataNotModifiedTest : public sh::ShaderCompileTreeTest
{
  public:
    GlFragDataNotModifiedTest() {}

  protected:
    void initResources(ShBuiltInResources *resources) override
    {
        resources->MaxDrawBuffers   = 4;
        resources->EXT_draw_buffers = 1;
    }

    ::GLenum getShaderType() const override { return GL_FRAGMENT_SHADER; }
    ShShaderSpec getShaderSpec() const override { return SH_GLES2_SPEC; }
};

// Test a bug where we could modify the value of a builtin variable.
TEST_F(GlFragDataNotModifiedTest, BuiltinRewritingBug)
{
    const std::string &shaderString =
        "#extension GL_EXT_draw_buffers : require\n"
        "precision mediump float;\n"
        "void main() {\n"
        "    gl_FragData[gl_MaxDrawBuffers] = vec4(0.0);\n"
        "}";

    if (compile(shaderString))
    {
        FAIL() << "Shader compilation succeeded, expecting failure\n";
    }
    if (compile(shaderString))
    {
        FAIL() << "Shader compilation succeeded, expecting failure\n";
    }
}

}  // anonymous namespace
