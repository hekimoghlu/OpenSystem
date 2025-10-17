/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include "GPUTextureFormat.h"
#include "WebGPURenderPassLayout.h"
#include <optional>
#include <wtf/Vector.h>

namespace WebCore {

struct GPURenderPassLayout : public GPUObjectDescriptorBase {
    WebGPU::RenderPassLayout convertToBacking() const
    {
        return {
            { label },
            colorFormats.map([](auto& colorFormat) -> std::optional<WebGPU::TextureFormat> {
                if (colorFormat)
                    return WebCore::convertToBacking(*colorFormat);
                return std::nullopt;
            }),
            depthStencilFormat ? std::optional { WebCore::convertToBacking(*depthStencilFormat) } : std::nullopt,
            sampleCount,
        };
    }

    Vector<std::optional<GPUTextureFormat>> colorFormats;
    std::optional<GPUTextureFormat> depthStencilFormat;
    GPUSize32 sampleCount { 1 };
};

}
