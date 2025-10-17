/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef COMPILER_TRANSLATOR_HLSL_TRANSLATORHLSL_H_
#define COMPILER_TRANSLATOR_HLSL_TRANSLATORHLSL_H_

#include "compiler/translator/Compiler.h"

namespace sh
{

class TranslatorHLSL : public TCompiler
{
  public:
    TranslatorHLSL(sh::GLenum type, ShShaderSpec spec, ShShaderOutput output);
    TranslatorHLSL *getAsTranslatorHLSL() override { return this; }

    bool hasShaderStorageBlock(const std::string &interfaceBlockName) const;
    unsigned int getShaderStorageBlockRegister(const std::string &interfaceBlockName) const;

    bool hasUniformBlock(const std::string &interfaceBlockName) const;
    unsigned int getUniformBlockRegister(const std::string &interfaceBlockName) const;
    bool shouldUniformBlockUseStructuredBuffer(const std::string &uniformBlockName) const;
    const std::set<std::string> *getSlowCompilingUniformBlockSet() const;

    const std::map<std::string, unsigned int> *getUniformRegisterMap() const;
    unsigned int getReadonlyImage2DRegisterIndex() const;
    unsigned int getImage2DRegisterIndex() const;
    const std::set<std::string> *getUsedImage2DFunctionNames() const;

  protected:
    [[nodiscard]] bool translate(TIntermBlock *root,
                                 const ShCompileOptions &compileOptions,
                                 PerformanceDiagnostics *perfDiagnostics) override;
    bool shouldFlattenPragmaStdglInvariantAll() override;

    std::map<std::string, unsigned int> mShaderStorageBlockRegisterMap;
    std::map<std::string, unsigned int> mUniformBlockRegisterMap;
    std::map<std::string, bool> mUniformBlockUseStructuredBufferMap;
    std::map<std::string, unsigned int> mUniformRegisterMap;
    unsigned int mReadonlyImage2DRegisterIndex;
    unsigned int mImage2DRegisterIndex;
    std::set<std::string> mUsedImage2DFunctionNames;
    std::map<int, const TInterfaceBlock *> mUniformBlockOptimizedMap;
    std::set<std::string> mSlowCompilingUniformBlockSet;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_HLSL_TRANSLATORHLSL_H_
