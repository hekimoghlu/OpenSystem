/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 5, 2025.
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

#include "GPUBufferBindingLayout.h"
#include "GPUExternalTextureBindingLayout.h"
#include "GPUIntegralTypes.h"
#include "GPUSamplerBindingLayout.h"
#include "GPUShaderStage.h"
#include "GPUStorageTextureBindingLayout.h"
#include "GPUTextureBindingLayout.h"
#include "WebGPUBindGroupLayoutEntry.h"
#include <optional>

namespace WebCore {

struct GPUBindGroupLayoutEntry {
    WebGPU::BindGroupLayoutEntry convertToBacking() const
    {
        return {
            binding,
            convertShaderStageFlagsToBacking(visibility),
            buffer ? std::optional { buffer->convertToBacking() } : std::nullopt,
            sampler ? std::optional { sampler->convertToBacking() } : std::nullopt,
            texture ? std::optional { texture->convertToBacking() } : std::nullopt,
            storageTexture ? std::optional { storageTexture->convertToBacking() } : std::nullopt,
            externalTexture ? std::optional { externalTexture->convertToBacking() } : std::nullopt,
        };
    }

    GPUIndex32 binding { 0 };
    GPUShaderStageFlags visibility { 0 };

    std::optional<GPUBufferBindingLayout> buffer;
    std::optional<GPUSamplerBindingLayout> sampler;
    std::optional<GPUTextureBindingLayout> texture;
    std::optional<GPUStorageTextureBindingLayout> storageTexture;
    std::optional<GPUExternalTextureBindingLayout> externalTexture;
};

}
