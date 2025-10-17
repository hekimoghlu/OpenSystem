/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 11, 2024.
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
#include "WebGPUBindGroupEntry.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUBindGroupEntry.h>
#include <WebCore/WebGPUExternalTexture.h>
#include <WebCore/WebGPUSampler.h>
#include <WebCore/WebGPUTexture.h>

namespace WebKit::WebGPU {

std::optional<BindGroupEntry> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::BindGroupEntry& bindGroupEntry)
{
    return WTF::switchOn(bindGroupEntry.resource, [&] (std::reference_wrapper<WebCore::WebGPU::Sampler> sampler) -> std::optional<BindGroupEntry> {
        auto identifier = convertToBacking(Ref { sampler.get() }.get());

        return { { bindGroupEntry.binding, { identifier }, identifier, BindingResourceType::Sampler } };
    }, [&] (std::reference_wrapper<WebCore::WebGPU::TextureView> textureView) -> std::optional<BindGroupEntry> {
        auto identifier = convertToBacking(Ref { textureView.get() }.get());

        return { { bindGroupEntry.binding, { identifier }, identifier, BindingResourceType::TextureView } };
    }, [&] (const auto& bufferBinding) -> std::optional<BindGroupEntry> {
        auto convertedBufferBinding = convertToBacking(bufferBinding);
        if (!convertedBufferBinding)
            return std::nullopt;

        return { { bindGroupEntry.binding, WTFMove(*convertedBufferBinding), convertedBufferBinding->buffer, BindingResourceType::BufferBinding } };
    }, [&] (std::reference_wrapper<WebCore::WebGPU::ExternalTexture> externalTexture) -> std::optional<BindGroupEntry> {
        auto identifier = convertToBacking(Ref { externalTexture.get() }.get());

        return { { bindGroupEntry.binding, { identifier }, identifier, BindingResourceType::ExternalTexture } };
    });
}

std::optional<WebCore::WebGPU::BindGroupEntry> ConvertFromBackingContext::convertFromBacking(const BindGroupEntry& bindGroupEntry)
{
    switch (bindGroupEntry.type) {
    case BindingResourceType::Sampler: {
        WeakPtr sampler = convertSamplerFromBacking(bindGroupEntry.identifier);
        if (!sampler)
            return std::nullopt;
        return { { bindGroupEntry.binding, { *sampler } } };
    }
    case BindingResourceType::TextureView: {
        WeakPtr textureView = convertTextureViewFromBacking(bindGroupEntry.identifier);
        if (!textureView)
            return std::nullopt;
        return { { bindGroupEntry.binding, { *textureView } } };
    }
    case BindingResourceType::BufferBinding: {
        auto bufferBinding = convertFromBacking(bindGroupEntry.bufferBinding);
        if (!bufferBinding)
            return std::nullopt;
        return { { bindGroupEntry.binding, { *bufferBinding } } };
    }
    case BindingResourceType::ExternalTexture: {
        auto externalTexture = convertExternalTextureFromBacking(bindGroupEntry.identifier);
        if (!externalTexture.get())
            return std::nullopt;
        return { { bindGroupEntry.binding, { *externalTexture.get() } } };
    }
    }

    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
