/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
// ShaderExtensionTest.cpp:
//   Utilities for testing that GLSL extension pragma and changing the extension flag in compiler
//   resources has the correct effect.
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "gtest/gtest.h"

using testing::Combine;
using testing::make_tuple;
using testing::Values;

namespace sh
{

const char ESSLVersion100[] = "#version 100\n";
const char ESSLVersion300[] = "#version 300 es\n";
const char ESSLVersion310[] = "#version 310 es\n";

class ShaderExtensionTest
    : public testing::TestWithParam<testing::tuple<ShShaderSpec, const char *, const char *>>
{
  protected:
    void SetUp() override
    {
        sh::InitBuiltInResources(&mResources);
        mCompiler                  = nullptr;
        mCompileOptions            = {};
        mCompileOptions.objectCode = true;
    }

    void TearDown() override { DestroyCompiler(); }
    void DestroyCompiler()
    {
        if (mCompiler)
        {
            sh::Destruct(mCompiler);
            mCompiler = nullptr;
        }
    }

    void InitializeCompiler()
    {
        DestroyCompiler();
        mCompiler = sh::ConstructCompiler(GL_FRAGMENT_SHADER, testing::get<0>(GetParam()),
                                          SH_GLSL_COMPATIBILITY_OUTPUT, &mResources);
        ASSERT_TRUE(mCompiler != nullptr) << "Compiler could not be constructed.";
    }

    testing::AssertionResult TestShaderCompile(const char *pragma)
    {
        return TestShaderCompile(testing::get<1>(GetParam()),  // Version.
                                 pragma,
                                 testing::get<2>(GetParam())  // Shader.
        );
    }

    testing::AssertionResult TestShaderCompile(const char *version,
                                               const char *pragma,
                                               const char *shader)
    {
        const char *shaderStrings[] = {version, pragma, shader};
        bool success                = sh::Compile(mCompiler, shaderStrings, 3, mCompileOptions);
        if (success)
        {
            return ::testing::AssertionSuccess() << "Compilation success";
        }
        return ::testing::AssertionFailure() << sh::GetInfoLog(mCompiler);
    }

  protected:
    ShBuiltInResources mResources;
    ShHandle mCompiler;
    ShCompileOptions mCompileOptions;
};

}  // namespace sh
