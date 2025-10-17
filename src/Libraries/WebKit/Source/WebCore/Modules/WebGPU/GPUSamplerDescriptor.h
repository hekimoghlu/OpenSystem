/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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

#include "GPUAddressMode.h"
#include "GPUCompareFunction.h"
#include "GPUFilterMode.h"
#include "GPUMipmapFilterMode.h"
#include "GPUObjectDescriptorBase.h"
#include "WebGPUSamplerDescriptor.h"
#include <cstdint>
#include <optional>

namespace WebCore {

struct GPUSamplerDescriptor : public GPUObjectDescriptorBase {
    WebGPU::SamplerDescriptor convertToBacking() const
    {
        return {
            { label },
            WebCore::convertToBacking(addressModeU),
            WebCore::convertToBacking(addressModeV),
            WebCore::convertToBacking(addressModeW),
            WebCore::convertToBacking(magFilter),
            WebCore::convertToBacking(minFilter),
            WebCore::convertToBacking(mipmapFilter),
            lodMinClamp,
            lodMaxClamp,
            compare ? std::optional { WebCore::convertToBacking(*compare) } : std::nullopt,
            maxAnisotropy,
        };
    }

    GPUAddressMode addressModeU { GPUAddressMode::ClampToEdge };
    GPUAddressMode addressModeV { GPUAddressMode::ClampToEdge };
    GPUAddressMode addressModeW { GPUAddressMode::ClampToEdge };
    GPUFilterMode magFilter { GPUFilterMode::Nearest };
    GPUFilterMode minFilter { GPUFilterMode::Nearest };
    GPUMipmapFilterMode mipmapFilter { GPUMipmapFilterMode::Nearest };
    float lodMinClamp { 0 };
    float lodMaxClamp { 32 };
    std::optional<GPUCompareFunction> compare;
    uint16_t maxAnisotropy { 1 };
};

}
