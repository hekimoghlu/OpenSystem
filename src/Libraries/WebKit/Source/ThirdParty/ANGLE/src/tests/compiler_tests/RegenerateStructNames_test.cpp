/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
// RegenerateStructNames_test.cpp:
//   Tests for regenerating struct names.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

class RegenerateStructNamesTest : public MatchOutputCodeTest
{
  public:
    RegenerateStructNamesTest() : MatchOutputCodeTest(GL_FRAGMENT_SHADER, SH_ESSL_OUTPUT)
    {
        ShCompileOptions defaultCompileOptions      = {};
        defaultCompileOptions.regenerateStructNames = true;
        setDefaultCompileOptions(defaultCompileOptions);
    }
};

// Test that a struct defined in a function scope is renamed. The global struct that's used as a
// type of a uniform cannot be renamed.
TEST_F(RegenerateStructNamesTest, GlobalStructAndLocalStructWithTheSameName)
{
    const std::string &shaderString =
        R"(precision mediump float;

        struct myStruct
        {
            float foo;
        };

        uniform myStruct us;

        void main()
        {
            struct myStruct
            {
                vec2 bar;
            };
            myStruct scoped;
            scoped.bar = vec2(1.0, 2.0) * us.foo;
            gl_FragColor = vec4(scoped.bar, 0.0, 1.0);
        })";
    compile(shaderString);
    EXPECT_TRUE(foundInCode("struct _umyStruct"));
    EXPECT_TRUE(foundInCode("struct _u_webgl_struct_"));
}

// Test that a nameless struct is handled gracefully.
TEST_F(RegenerateStructNamesTest, NamelessStruct)
{
    const std::string &shaderString =
        R"(precision mediump float;

        uniform float u;

        void main()
        {
            struct
            {
                vec2 bar;
            } scoped;
            scoped.bar = vec2(1.0, 2.0) * u;
            gl_FragColor = vec4(scoped.bar, 0.0, 1.0);
        })";
    compile(shaderString);
    EXPECT_TRUE(foundInCode("struct"));
}
