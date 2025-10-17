/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 30, 2024.
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
#include "WebGPURenderPipelineDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPURenderPipelineDescriptor.h>

namespace WebKit::WebGPU {

std::optional<RenderPipelineDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderPipelineDescriptor& renderPipelineDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::PipelineDescriptorBase&>(renderPipelineDescriptor));
    if (!base)
        return std::nullopt;

    auto vertex = convertToBacking(renderPipelineDescriptor.vertex);
    if (!vertex)
        return std::nullopt;

    std::optional<PrimitiveState> primitive;
    if (renderPipelineDescriptor.primitive) {
        primitive = convertToBacking(*renderPipelineDescriptor.primitive);
        if (!primitive)
            return std::nullopt;
    }

    std::optional<DepthStencilState> depthStencil;
    if (renderPipelineDescriptor.depthStencil) {
        depthStencil = convertToBacking(*renderPipelineDescriptor.depthStencil);
        if (!depthStencil)
            return std::nullopt;
    }

    std::optional<MultisampleState> multisample;
    if (renderPipelineDescriptor.multisample) {
        multisample = convertToBacking(*renderPipelineDescriptor.multisample);
        if (!multisample)
            return std::nullopt;
    }

    std::optional<FragmentState> fragment;
    if (renderPipelineDescriptor.fragment) {
        fragment = convertToBacking(*renderPipelineDescriptor.fragment);
        if (!fragment)
            return std::nullopt;
    }
    if (renderPipelineDescriptor.fragment && !fragment)
        return std::nullopt;

    return { { WTFMove(*base), WTFMove(*vertex), WTFMove(primitive), WTFMove(depthStencil), WTFMove(multisample), WTFMove(fragment) } };
}

std::optional<WebCore::WebGPU::RenderPipelineDescriptor> ConvertFromBackingContext::convertFromBacking(const RenderPipelineDescriptor& renderPipelineDescriptor)
{
    auto base = convertFromBacking(static_cast<const PipelineDescriptorBase&>(renderPipelineDescriptor));
    if (!base)
        return std::nullopt;

    auto vertex = convertFromBacking(renderPipelineDescriptor.vertex);
    if (!vertex)
        return std::nullopt;

    std::optional<WebCore::WebGPU::PrimitiveState> primitive;
    if (renderPipelineDescriptor.primitive) {
        primitive = convertFromBacking(*renderPipelineDescriptor.primitive);
        if (!primitive)
            return std::nullopt;
    }

    std::optional<WebCore::WebGPU::DepthStencilState> depthStencil;
    if (renderPipelineDescriptor.depthStencil) {
        depthStencil = convertFromBacking(*renderPipelineDescriptor.depthStencil);
        if (!depthStencil)
            return std::nullopt;
    }

    std::optional<WebCore::WebGPU::MultisampleState> multisample;
    if (renderPipelineDescriptor.multisample) {
        multisample = convertFromBacking(*renderPipelineDescriptor.multisample);
        if (!multisample)
            return std::nullopt;
    }

    auto fragment = ([&] () -> std::optional<WebCore::WebGPU::FragmentState> {
        if (renderPipelineDescriptor.fragment)
            return convertFromBacking(*renderPipelineDescriptor.fragment);
        return std::nullopt;
    })();
    if (renderPipelineDescriptor.fragment && !fragment)
        return std::nullopt;

    return { { WTFMove(*base), WTFMove(*vertex), WTFMove(primitive), WTFMove(depthStencil), WTFMove(multisample), WTFMove(fragment) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
