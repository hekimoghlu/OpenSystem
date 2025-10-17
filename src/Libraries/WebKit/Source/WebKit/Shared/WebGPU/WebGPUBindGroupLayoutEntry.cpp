/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include "config.h"
#include "WebGPUBindGroupLayoutEntry.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUBindGroupLayoutEntry.h>

namespace WebKit::WebGPU {

std::optional<BindGroupLayoutEntry> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::BindGroupLayoutEntry& bindGroupLayoutEntry)
{
    std::optional<BufferBindingLayout> buffer;
    if (bindGroupLayoutEntry.buffer) {
        buffer = convertToBacking(*bindGroupLayoutEntry.buffer);
        if (!buffer)
            return std::nullopt;
    }

    std::optional<SamplerBindingLayout> sampler;
    if (bindGroupLayoutEntry.sampler) {
        sampler = convertToBacking(*bindGroupLayoutEntry.sampler);
        if (!sampler)
            return std::nullopt;
    }

    std::optional<TextureBindingLayout> texture;
    if (bindGroupLayoutEntry.texture) {
        texture = convertToBacking(*bindGroupLayoutEntry.texture);
        if (!texture)
            return std::nullopt;
    }

    std::optional<StorageTextureBindingLayout> storageTexture;
    if (bindGroupLayoutEntry.storageTexture) {
        storageTexture = convertToBacking(*bindGroupLayoutEntry.storageTexture);
        if (!storageTexture)
            return std::nullopt;
    }

    std::optional<ExternalTextureBindingLayout> externalTexture;
    if (bindGroupLayoutEntry.externalTexture) {
        externalTexture = convertToBacking(*bindGroupLayoutEntry.externalTexture);
        if (!externalTexture)
            return std::nullopt;
    }

    return { { bindGroupLayoutEntry.binding, bindGroupLayoutEntry.visibility, WTFMove(buffer), WTFMove(sampler), WTFMove(texture), WTFMove(storageTexture), WTFMove(externalTexture) } };
}

std::optional<WebCore::WebGPU::BindGroupLayoutEntry> ConvertFromBackingContext::convertFromBacking(const BindGroupLayoutEntry& bindGroupLayoutEntry)
{
    std::optional<WebCore::WebGPU::BufferBindingLayout> buffer;
    if (bindGroupLayoutEntry.buffer) {
        buffer = convertFromBacking(*bindGroupLayoutEntry.buffer);
        if (!buffer)
            return std::nullopt;
    }

    std::optional<WebCore::WebGPU::SamplerBindingLayout> sampler;
    if (bindGroupLayoutEntry.sampler) {
        sampler = convertFromBacking(*bindGroupLayoutEntry.sampler);
        if (!sampler)
            return std::nullopt;
    }

    std::optional<WebCore::WebGPU::TextureBindingLayout> texture;
    if (bindGroupLayoutEntry.texture) {
        texture = convertFromBacking(*bindGroupLayoutEntry.texture);
        if (!texture)
            return std::nullopt;
    }

    std::optional<WebCore::WebGPU::StorageTextureBindingLayout> storageTexture;
    if (bindGroupLayoutEntry.storageTexture) {
        storageTexture = convertFromBacking(*bindGroupLayoutEntry.storageTexture);
        if (!storageTexture)
            return std::nullopt;
    }

    std::optional<WebCore::WebGPU::ExternalTextureBindingLayout> externalTexture;
    if (bindGroupLayoutEntry.externalTexture) {
        externalTexture = convertFromBacking(*bindGroupLayoutEntry.externalTexture);
        if (!externalTexture)
            return std::nullopt;
    }

    return { { bindGroupLayoutEntry.binding, bindGroupLayoutEntry.visibility, WTFMove(buffer), WTFMove(sampler), WTFMove(texture), WTFMove(storageTexture), WTFMove(externalTexture) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
