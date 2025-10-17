/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// PruneEmptyDeclarations_test.cpp:
//   Tests for pruning empty declarations.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

namespace
{

class PruneEmptyDeclarationsTest : public MatchOutputCodeTest
{
  public:
    PruneEmptyDeclarationsTest()
        : MatchOutputCodeTest(GL_VERTEX_SHADER, SH_GLSL_COMPATIBILITY_OUTPUT)
    {}
};

TEST_F(PruneEmptyDeclarationsTest, EmptyDeclarationStartsDeclaratorList)
{
    const std::string shaderString =
        "precision mediump float;\n"
        "uniform float u;\n"
        "void main()\n"
        "{\n"
        "   float, f;\n"
        "   gl_Position = vec4(u * f);\n"
        "}\n";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("float _uf"));
    ASSERT_TRUE(notFoundInCode("float, _uf"));
    ASSERT_TRUE(notFoundInCode("float, f"));
    ASSERT_TRUE(notFoundInCode("float _u, _uf"));
}

TEST_F(PruneEmptyDeclarationsTest, EmptyStructDeclarationWithQualifiers)
{
    const std::string shaderString =
        "precision mediump float;\n"
        "const struct S { float f; };\n"
        "uniform S s;"
        "void main()\n"
        "{\n"
        "   gl_Position = vec4(s.f);\n"
        "}\n";
    compile(shaderString);
    ASSERT_TRUE(foundInCode("struct _uS"));
    ASSERT_TRUE(foundInCode("uniform _uS"));
    ASSERT_TRUE(notFoundInCode("const struct _uS"));
}

}  // namespace
