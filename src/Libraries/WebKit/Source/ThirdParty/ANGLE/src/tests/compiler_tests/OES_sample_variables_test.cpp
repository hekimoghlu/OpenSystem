/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// OES_sample_variables_test.cpp:
//   Test for OES_sample_variables
//

#include "tests/test_utils/ShaderExtensionTest.h"

namespace
{
const char OESPragma[] = "#extension GL_OES_sample_variables : require\n";

// Shader using gl_SampleMask with non-constant index
// This shader is in the deqp test
// (functional_shaders_sample_variables_sample_mask_discard_half_per_sample_default_framebuffer)
const char ESSL310_GLSampleMaskShader[] =
    R"(
    layout(location = 0) out mediump vec4 fragColor;
    void main (void)
    {
        for (int i = 0; i < gl_SampleMask.length(); ++i)
                gl_SampleMask[i] = int(0xAAAAAAAA);

        // force per-sample shading
        highp float blue = float(gl_SampleID);

        fragColor = vec4(0.0, 1.0, blue, 1.0);
    })";

// Shader using gl_SampleMask with non-constant index
// This shader is based on the deqp test on below
// (functional_shaders_sample_variables_sample_mask_in_bit_count_per_sample_multisample_texture_2)
const char ESSL310_GLSampleMaskInShader[] =
    R"(
    layout(location = 0) out mediump vec4 fragColor;
    void main (void)
    {
        mediump int maskBitCount = 0;
        for (int j = 0; j < gl_SampleMaskIn.length(); ++j)
        {
            for (int i = 0; i < 32; ++i)
            {
                if (((gl_SampleMaskIn[j] >> i) & 0x01) == 0x01)
                {
                    ++maskBitCount;
                }
            }
        }

        // force per-sample shading
        highp float blue = float(gl_SampleID);

        if (maskBitCount != 1)
            fragColor = vec4(1.0, 0.0, blue, 1.0);
        else
            fragColor = vec4(0.0, 1.0, blue, 1.0);
    })";

class OESSampleVariablesTest : public sh::ShaderExtensionTest
{
  public:
    void InitializeCompiler() { InitializeCompiler(SH_GLSL_450_CORE_OUTPUT); }
    void InitializeCompiler(ShShaderOutput shaderOutputType)
    {
        DestroyCompiler();

        mCompiler = sh::ConstructCompiler(GL_FRAGMENT_SHADER, testing::get<0>(GetParam()),
                                          shaderOutputType, &mResources);
        ASSERT_TRUE(mCompiler != nullptr) << "Compiler could not be constructed.";
    }

    testing::AssertionResult TestShaderCompile(const char *pragma)
    {
        const char *shaderStrings[] = {testing::get<1>(GetParam()), pragma,
                                       testing::get<2>(GetParam())};

        ShCompileOptions compileOptions = {};
        compileOptions.objectCode       = true;

        bool success = sh::Compile(mCompiler, shaderStrings, 3, compileOptions);
        if (success)
        {
            return ::testing::AssertionSuccess() << "Compilation success";
        }
        return ::testing::AssertionFailure() << sh::GetInfoLog(mCompiler);
    }
};

// GLES3 needs OES_sample_variables extension
class OESSampleVariablesTestES31 : public OESSampleVariablesTest
{};

// Extension flag is required to compile properly. Expect failure when it is
// not present.
TEST_P(OESSampleVariablesTestES31, CompileFailsWithoutExtension)
{
    mResources.OES_sample_variables = 0;
    InitializeCompiler();
    EXPECT_FALSE(TestShaderCompile(OESPragma));
}

// Extension directive is required to compile properly. Expect failure when
// it is not present.
TEST_P(OESSampleVariablesTestES31, CompileFailsWithExtensionWithoutPragma)
{
    mResources.OES_sample_variables = 1;
    InitializeCompiler();
    EXPECT_FALSE(TestShaderCompile(""));
}

// With extension flag and extension directive, compiling succeeds.
// Also test that the extension directive state is reset correctly.
#ifdef ANGLE_ENABLE_VULKAN
TEST_P(OESSampleVariablesTestES31, CompileSucceedsWithExtensionAndPragmaOnVulkan)
{
    mResources.OES_sample_variables = 1;
    InitializeCompiler(SH_SPIRV_VULKAN_OUTPUT);
    EXPECT_TRUE(TestShaderCompile(OESPragma));
    // Test reset functionality.
    EXPECT_FALSE(TestShaderCompile(""));
    EXPECT_TRUE(TestShaderCompile(OESPragma));
}
#endif

INSTANTIATE_TEST_SUITE_P(CorrectESSL310Shaders,
                         OESSampleVariablesTestES31,
                         Combine(Values(SH_GLES3_1_SPEC),
                                 Values(sh::ESSLVersion310),
                                 Values(ESSL310_GLSampleMaskShader, ESSL310_GLSampleMaskInShader)));

}  // anonymous namespace
