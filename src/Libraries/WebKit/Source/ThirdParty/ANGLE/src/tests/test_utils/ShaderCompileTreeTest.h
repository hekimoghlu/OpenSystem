/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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
// ShaderCompileTreeTest.h:
//   Test that shader validation results in the correct compile status.
//

#ifndef TESTS_TEST_UTILS_SHADER_COMPILE_TREE_TEST_H_
#define TESTS_TEST_UTILS_SHADER_COMPILE_TREE_TEST_H_

#include "GLSLANG/ShaderLang.h"
#include "angle_gl.h"
#include "compiler/translator/PoolAlloc.h"
#include "gtest/gtest.h"

namespace sh
{

class TIntermBlock;
class TIntermNode;
class TranslatorESSL;

class ShaderCompileTreeTest : public testing::Test
{
  public:
    ShaderCompileTreeTest() : mCompileOptions{} {}

  protected:
    void SetUp() override;

    void TearDown() override;

    // Return true when compilation succeeds
    bool compile(const std::string &shaderString);
    void compileAssumeSuccess(const std::string &shaderString);

    bool hasWarning() const;

    const std::vector<sh::ShaderVariable> &getUniforms() const;
    const std::vector<sh::ShaderVariable> &getAttributes() const;

    virtual void initResources(ShBuiltInResources *resources) {}
    virtual ::GLenum getShaderType() const     = 0;
    virtual ShShaderSpec getShaderSpec() const = 0;

    std::string mInfoLog;
    ShCompileOptions mCompileOptions;

    TIntermBlock *mASTRoot;

  private:
    TranslatorESSL *mTranslator;

    angle::PoolAllocator mAllocator;
};

// Returns true if the node is some kind of a zero node - either constructor or a constant union
// node.
bool IsZero(TIntermNode *node);

}  // namespace sh

#endif  // TESTS_TEST_UTILS_SHADER_COMPILE_TREE_TEST_H_
