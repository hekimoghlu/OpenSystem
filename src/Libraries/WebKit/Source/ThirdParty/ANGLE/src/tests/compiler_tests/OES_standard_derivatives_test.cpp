/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
// OES_standard_derivatives_test.cpp:
//   Test for OES_standard_derivatives
//

#include "tests/test_utils/ShaderExtensionTest.h"

using OESStandardDerivativesTest = sh::ShaderExtensionTest;

namespace
{
const char OESPragma[] = "#extension GL_OES_standard_derivatives : require\n";

// Shader calling dFdx()
const char ESSL100_DfdxShader[] =
    R"(
    precision mediump float;
    varying float x;

    void main()
    {
        gl_FragColor = vec4(dFdx(x));
     })";

// Shader calling dFdy()
const char ESSL100_DfdyShader[] =
    R"(
    precision mediump float;
    varying float x;

    void main()
    {
        gl_FragColor = vec4(dFdy(x));
    })";

// Shader calling fwidth()
const char ESSL100_FwidthShader[] =
    R"(
    precision mediump float;
    varying float x;

    void main()
    {
        gl_FragColor = vec4(fwidth(x));
    })";

// Extension flag is required to compile properly. Expect failure when it is
// not present.
TEST_P(OESStandardDerivativesTest, CompileFailsWithoutExtension)
{
    mResources.OES_standard_derivatives = 0;
    InitializeCompiler();
    EXPECT_FALSE(TestShaderCompile(OESPragma));
}

// Extension directive is required to compile properly. Expect failure when
// it is not present.
TEST_P(OESStandardDerivativesTest, CompileFailsWithExtensionWithoutPragma)
{
    mResources.OES_standard_derivatives = 1;
    InitializeCompiler();
    EXPECT_FALSE(TestShaderCompile(""));
}

// With extension flag and extension directive, compiling succeeds.
// Also test that the extension directive state is reset correctly.
TEST_P(OESStandardDerivativesTest, CompileSucceedsWithExtensionAndPragma)
{
    mResources.OES_standard_derivatives = 1;
    InitializeCompiler();
    EXPECT_TRUE(TestShaderCompile(OESPragma));
    // Test reset functionality.
    EXPECT_FALSE(TestShaderCompile(""));
    EXPECT_TRUE(TestShaderCompile(OESPragma));
}

// The SL #version 100 shaders that are correct work similarly
// in both GL2 and GL3, with and without the version string.
INSTANTIATE_TEST_SUITE_P(
    CorrectESSL100Shaders,
    OESStandardDerivativesTest,
    Combine(Values(SH_GLES2_SPEC),
            Values(sh::ESSLVersion100),
            Values(ESSL100_DfdxShader, ESSL100_DfdyShader, ESSL100_FwidthShader)));

}  // anonymous namespace
