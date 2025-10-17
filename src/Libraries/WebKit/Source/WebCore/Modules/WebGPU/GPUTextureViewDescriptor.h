/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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

#include "GPUIntegralTypes.h"
#include "GPUObjectDescriptorBase.h"
#include "GPUTextureAspect.h"
#include "GPUTextureFormat.h"
#include "GPUTextureViewDimension.h"
#include "WebGPUTextureViewDescriptor.h"
#include <optional>

namespace WebCore {

struct GPUTextureViewDescriptor : public GPUObjectDescriptorBase {
    WebGPU::TextureViewDescriptor convertToBacking() const
    {
        return {
            { label },
            format ? std::optional { WebCore::convertToBacking(*format) } : std::nullopt,
            dimension ? std::optional { WebCore::convertToBacking(*dimension) } : std::nullopt,
            WebCore::convertToBacking(aspect),
            baseMipLevel,
            mipLevelCount,
            baseArrayLayer,
            arrayLayerCount,
        };
    }

    std::optional<GPUTextureFormat> format;
    std::optional<GPUTextureViewDimension> dimension;
    GPUTextureAspect aspect { GPUTextureAspect::All };
    GPUIntegerCoordinate baseMipLevel { 0 };
    std::optional<GPUIntegerCoordinate> mipLevelCount;
    GPUIntegerCoordinate baseArrayLayer { 0 };
    std::optional<GPUIntegerCoordinate> arrayLayerCount;
};

}
