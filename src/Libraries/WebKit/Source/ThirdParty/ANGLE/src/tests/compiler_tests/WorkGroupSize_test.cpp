/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WorkGroupSize_test.cpp:
// tests for local group size in a compute shader
//

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "compiler/translator/glsl/TranslatorESSL.h"
#include "gtest/gtest.h"
#include "tests/test_utils/compiler_test.h"

using namespace sh;

class WorkGroupSizeTest : public testing::Test
{
  public:
    WorkGroupSizeTest() {}

  protected:
    void SetUp() override
    {
        ShBuiltInResources resources;
        InitBuiltInResources(&resources);

        mTranslator = new TranslatorESSL(GL_COMPUTE_SHADER, SH_GLES3_1_SPEC);
        ASSERT_TRUE(mTranslator->Init(resources));
    }

    void TearDown() override { SafeDelete(mTranslator); }

    // Return true when compilation succeeds
    bool compile(const std::string &shaderString)
    {
        ShCompileOptions compileOptions = {};
        compileOptions.intermediateTree = true;

        const char *shaderStrings[] = {shaderString.c_str()};
        bool status                 = mTranslator->compile(shaderStrings, 1, compileOptions);
        TInfoSink &infoSink         = mTranslator->getInfoSink();
        mInfoLog                    = infoSink.info.c_str();
        return status;
    }

  protected:
    std::string mInfoLog;
    TranslatorESSL *mTranslator = nullptr;
};

// checks whether compiler has parsed the local size layout qualifiers qcorrectly
TEST_F(WorkGroupSizeTest, OnlyLocalSizeXSpecified)
{
    const std::string &shaderString =
        "#version 310 es\n"
        "layout(local_size_x=5) in;\n"
        "void main() {\n"
        "}\n";

    compile(shaderString);

    const WorkGroupSize &localSize = mTranslator->getComputeShaderLocalSize();
    ASSERT_EQ(5, localSize[0]);
    ASSERT_EQ(1, localSize[1]);
    ASSERT_EQ(1, localSize[2]);
}

// checks whether compiler has parsed the local size layout qualifiers qcorrectly
TEST_F(WorkGroupSizeTest, LocalSizeXandZ)
{
    const std::string &shaderString =
        "#version 310 es\n"
        "layout(local_size_x=5, local_size_z=10) in;\n"
        "void main() {\n"
        "}\n";

    compile(shaderString);

    const WorkGroupSize &localSize = mTranslator->getComputeShaderLocalSize();
    ASSERT_EQ(5, localSize[0]);
    ASSERT_EQ(1, localSize[1]);
    ASSERT_EQ(10, localSize[2]);
}

// checks whether compiler has parsed the local size layout qualifiers qcorrectly
TEST_F(WorkGroupSizeTest, LocalSizeAll)
{
    const std::string &shaderString =
        "#version 310 es\n"
        "layout(local_size_x=5, local_size_z=10, local_size_y=15) in;\n"
        "void main() {\n"
        "}\n";

    compile(shaderString);

    const WorkGroupSize &localSize = mTranslator->getComputeShaderLocalSize();
    ASSERT_EQ(5, localSize[0]);
    ASSERT_EQ(15, localSize[1]);
    ASSERT_EQ(10, localSize[2]);
}
