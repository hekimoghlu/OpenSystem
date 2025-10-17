/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// RewriteDoWhile_test.cpp:
//   Tests that the RewriteDoWhile AST transform works correctly.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/ShaderCompileTreeTest.h"

using namespace sh;

class RewriteDoWhileCrashTest : public ShaderCompileTreeTest
{
  public:
    RewriteDoWhileCrashTest() {}

  protected:
    ::GLenum getShaderType() const override { return GL_FRAGMENT_SHADER; }
    ShShaderSpec getShaderSpec() const override { return SH_GLES3_1_SPEC; }

    void SetUp() override
    {
        mCompileOptions.rewriteDoWhileLoops = true;
        ShaderCompileTreeTest::SetUp();
    }
};

// Make sure that the RewriteDoWhile step doesn't crash. Regression test.
TEST_F(RewriteDoWhileCrashTest, RunsSuccessfully)
{
    const std::string &shaderString =
        "#version 300 es\n"
        "precision mediump float;\n"
        "uniform int u;\n"
        "out vec4 my_FragColor;\n"
        "void main()\n"
        "{\n"
        "    int foo = 1;"
        "    do\n"
        "    {\n"
        "         foo *= u;\n"
        "    } while (foo < 8);\n"
        "    my_FragColor = vec4(foo) * 0.1;"
        "}\n";
    if (!compile(shaderString))
    {
        FAIL() << "Shader compilation failed, expecting success:\n" << mInfoLog;
    }
}
