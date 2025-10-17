/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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
#include "WebGPUTextureUsage.h"
#include <cstdint>
#include <wtf/RefCounted.h>

namespace WebCore {

using GPUTextureUsageFlags = uint32_t;
class GPUTextureUsage : public RefCounted<GPUTextureUsage> {
public:
    static constexpr GPUFlagsConstant COPY_SRC          = 0x01;
    static constexpr GPUFlagsConstant COPY_DST          = 0x02;
    static constexpr GPUFlagsConstant TEXTURE_BINDING   = 0x04;
    static constexpr GPUFlagsConstant STORAGE_BINDING   = 0x08;
    static constexpr GPUFlagsConstant RENDER_ATTACHMENT = 0x10;
};

inline WebGPU::TextureUsageFlags convertTextureUsageFlagsToBacking(GPUTextureUsageFlags textureUsageFlags)
{
    WebGPU::TextureUsageFlags result;
    if (textureUsageFlags & GPUTextureUsage::COPY_SRC)
        result.add(WebGPU::TextureUsage::CopySource);
    if (textureUsageFlags & GPUTextureUsage::COPY_DST)
        result.add(WebGPU::TextureUsage::CopyDestination);
    if (textureUsageFlags & GPUTextureUsage::TEXTURE_BINDING)
        result.add(WebGPU::TextureUsage::TextureBinding);
    if (textureUsageFlags & GPUTextureUsage::STORAGE_BINDING)
        result.add(WebGPU::TextureUsage::StorageBinding);
    if (textureUsageFlags & GPUTextureUsage::RENDER_ATTACHMENT)
        result.add(WebGPU::TextureUsage::RenderAttachment);
    return result;
}

}
