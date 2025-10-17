/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
// TranslatorSPIRV:
//   A set of transformations that prepare the AST to be compatible with GL_KHR_vulkan_glsl followed
//   by a pass that generates SPIR-V.
//   See: https://www.khronos.org/registry/vulkan/specs/misc/GL_KHR_vulkan_glsl.txt
//

#ifndef COMPILER_TRANSLATOR_SPIRV_TRANSLATORSPIRV_H_
#define COMPILER_TRANSLATOR_SPIRV_TRANSLATORSPIRV_H_

#include "common/hash_containers.h"
#include "compiler/translator/Compiler.h"

namespace sh
{

class TOutputVulkanGLSL;
class SpecConst;
class DriverUniform;

// An index -> TVariable map, tracking the declarated color input attachments, as well as TVariables
// for depth and stencil input attachments.
struct InputAttachmentMap
{
    TUnorderedMap<uint32_t, const TVariable *> color;
    const TVariable *depth   = nullptr;
    const TVariable *stencil = nullptr;
};

class TranslatorSPIRV final : public TCompiler
{
  public:
    TranslatorSPIRV(sh::GLenum type, ShShaderSpec spec);

    void assignSpirvId(TSymbolUniqueId uniqueId, uint32_t spirvId);

  protected:
    [[nodiscard]] bool translate(TIntermBlock *root,
                                 const ShCompileOptions &compileOptions,
                                 PerformanceDiagnostics *perfDiagnostics) override;
    bool shouldFlattenPragmaStdglInvariantAll() override;

    [[nodiscard]] bool translateImpl(TIntermBlock *root,
                                     const ShCompileOptions &compileOptions,
                                     PerformanceDiagnostics *perfDiagnostics,
                                     SpecConst *specConst,
                                     DriverUniform *driverUniforms);
    void assignInputAttachmentIds(const InputAttachmentMap &inputAttachmentMap);
    void assignSpirvIds(TIntermBlock *root);

    // A map from TSymbolUniqueId::mId to SPIR-V reserved ids.  Used by the SPIR-V generator to
    // quickly know when to use a reserved id and not have to resort to name matching.
    angle::HashMap<int, uint32_t> mUniqueToSpirvIdMap;
    uint32_t mFirstUnusedSpirvId;
};

}  // namespace sh

#endif  // COMPILER_TRANSLATOR_SPIRV_TRANSLATORSPIRV_H_
