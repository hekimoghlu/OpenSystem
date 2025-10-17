/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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
#pragma once

#include "WebGPUFeatureName.h"
#include <cstdint>

namespace WebCore {

enum class GPUFeatureName : uint8_t {
    DepthClipControl,
    Depth32floatStencil8,
    TextureCompressionBc,
    TextureCompressionBcSliced3d,
    TextureCompressionEtc2,
    TextureCompressionAstc,
    TextureCompressionAstcSliced3d,
    TimestampQuery,
    IndirectFirstInstance,
    ShaderF16,
    Rg11b10ufloatRenderable,
    Bgra8unormStorage,
    Float32Filterable,
    Float32Blendable,
    ClipDistances,
    DualSourceBlending,
    Float16Renderable,
    Float32Renderable,
};

inline WebGPU::FeatureName convertToBacking(GPUFeatureName featureName)
{
    switch (featureName) {
    case GPUFeatureName::DepthClipControl:
        return WebGPU::FeatureName::DepthClipControl;
    case GPUFeatureName::Depth32floatStencil8:
        return WebGPU::FeatureName::Depth32floatStencil8;
    case GPUFeatureName::TextureCompressionBc:
        return WebGPU::FeatureName::TextureCompressionBc;
    case GPUFeatureName::TextureCompressionBcSliced3d:
        return WebGPU::FeatureName::TextureCompressionBcSliced3d;
    case GPUFeatureName::TextureCompressionEtc2:
        return WebGPU::FeatureName::TextureCompressionEtc2;
    case GPUFeatureName::TextureCompressionAstc:
        return WebGPU::FeatureName::TextureCompressionAstc;
    case GPUFeatureName::TextureCompressionAstcSliced3d:
        return WebGPU::FeatureName::TextureCompressionAstcSliced3d;
    case GPUFeatureName::TimestampQuery:
        return WebGPU::FeatureName::TimestampQuery;
    case GPUFeatureName::IndirectFirstInstance:
        return WebGPU::FeatureName::IndirectFirstInstance;
    case GPUFeatureName::Bgra8unormStorage:
        return WebGPU::FeatureName::Bgra8unormStorage;
    case GPUFeatureName::ShaderF16:
        return WebGPU::FeatureName::ShaderF16;
    case GPUFeatureName::Rg11b10ufloatRenderable:
        return WebGPU::FeatureName::Rg11b10ufloatRenderable;
    case GPUFeatureName::Float32Filterable:
        return WebGPU::FeatureName::Float32Filterable;
    case GPUFeatureName::Float32Blendable:
        return WebGPU::FeatureName::Float32Blendable;
    case GPUFeatureName::Float16Renderable:
        return WebGPU::FeatureName::Float16Renderable;
    case GPUFeatureName::Float32Renderable:
        return WebGPU::FeatureName::Float32Renderable;
    case GPUFeatureName::DualSourceBlending:
        return WebGPU::FeatureName::DualSourceBlending;
    case GPUFeatureName::ClipDistances:
        return WebGPU::FeatureName::ClipDistances;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
