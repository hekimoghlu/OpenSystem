/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// mtl_msl_utils.h: Utilities to manipulate MSL.
//

#ifndef mtl_msl_utils_h
#define mtl_msl_utils_h

#include <memory>

#include "compiler/translator/msl/TranslatorMSL.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/ProgramImpl.h"
#include "libANGLE/renderer/metal/mtl_common.h"

namespace rx
{
struct CompiledShaderStateMtl : angle::NonCopyable
{
    sh::TranslatorMetalReflection translatorMetalReflection = {};
};
using SharedCompiledShaderStateMtl = std::shared_ptr<CompiledShaderStateMtl>;

namespace mtl
{
struct SamplerBinding
{
    uint32_t textureBinding = 0;
    uint32_t samplerBinding = 0;
};

struct TranslatedShaderInfo
{
    void reset();
    // Translated Metal source code
    std::shared_ptr<const std::string> metalShaderSource;
    // Metal library compiled from source code above. Used by ProgramMtl.
    AutoObjCPtr<id<MTLLibrary>> metalLibrary;
    std::array<SamplerBinding, kMaxGLSamplerBindings> actualSamplerBindings;
    std::array<int, kMaxShaderImages> actualImageBindings;
    std::array<uint32_t, kMaxGLUBOBindings> actualUBOBindings;
    std::array<uint32_t, kMaxShaderXFBs> actualXFBBindings;
    bool hasUBOArgumentBuffer;
    bool hasIsnanOrIsinf;
    bool hasInvariant;
};

void MSLGetShaderSource(const gl::ProgramState &programState,
                        const gl::ProgramLinkedResources &resources,
                        gl::ShaderMap<std::string> *shaderSourcesOut);

angle::Result MTLGetMSL(const angle::FeaturesMtl &features,
                        const gl::ProgramExecutable &executable,
                        const gl::ShaderMap<std::string> &shaderSources,
                        const gl::ShaderMap<SharedCompiledShaderStateMtl> &shadersState,
                        gl::ShaderMap<TranslatedShaderInfo> *mslShaderInfoOut);

// Get equivalent shadow compare mode that is used in translated msl shader.
uint MslGetShaderShadowCompareMode(GLenum mode, GLenum func);
}  // namespace mtl
}  // namespace rx

#endif /* mtl_msl_utils_h */
