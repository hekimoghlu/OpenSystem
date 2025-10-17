/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "WebGPURenderPassDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPURenderPassDescriptor.h>

namespace WebKit::WebGPU {

std::optional<RenderPassDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderPassDescriptor& renderPassDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ObjectDescriptorBase&>(renderPassDescriptor));
    if (!base)
        return std::nullopt;

    Vector<std::optional<RenderPassColorAttachment>> colorAttachments;
    colorAttachments.reserveInitialCapacity(renderPassDescriptor.colorAttachments.size());
    for (const auto& colorAttachment : renderPassDescriptor.colorAttachments) {
        if (colorAttachment) {
            auto backingColorAttachment = convertToBacking(*colorAttachment);
            if (!backingColorAttachment)
                return std::nullopt;
            colorAttachments.append(WTFMove(*backingColorAttachment));
        } else
            colorAttachments.append(std::nullopt);
    }

    std::optional<RenderPassDepthStencilAttachment> depthStencilAttachment;
    if (renderPassDescriptor.depthStencilAttachment) {
        depthStencilAttachment = convertToBacking(*renderPassDescriptor.depthStencilAttachment);
        if (!depthStencilAttachment)
            return std::nullopt;
    }

    std::optional<WebGPUIdentifier> occlusionQuerySet;
    if (renderPassDescriptor.occlusionQuerySet) {
        occlusionQuerySet = convertToBacking(*renderPassDescriptor.protectedOcclusionQuerySet());
        if (!occlusionQuerySet)
            return std::nullopt;
    }

    auto timestampWrites = renderPassDescriptor.timestampWrites ? convertToBacking(*renderPassDescriptor.timestampWrites) : std::nullopt;

    return { { WTFMove(*base), WTFMove(colorAttachments), WTFMove(depthStencilAttachment), occlusionQuerySet, WTFMove(timestampWrites), renderPassDescriptor.maxDrawCount } };
}

std::optional<WebCore::WebGPU::RenderPassDescriptor> ConvertFromBackingContext::convertFromBacking(const RenderPassDescriptor& renderPassDescriptor)
{
    auto base = convertFromBacking(static_cast<const ObjectDescriptorBase&>(renderPassDescriptor));
    if (!base)
        return std::nullopt;

    Vector<std::optional<WebCore::WebGPU::RenderPassColorAttachment>> colorAttachments;
    colorAttachments.reserveInitialCapacity(renderPassDescriptor.colorAttachments.size());
    for (const auto& backingColorAttachment : renderPassDescriptor.colorAttachments) {
        if (backingColorAttachment) {
            auto colorAttachment = convertFromBacking(*backingColorAttachment);
            if (!colorAttachment)
                return std::nullopt;
            colorAttachments.append(WTFMove(*colorAttachment));
        } else
            colorAttachments.append(std::nullopt);
    }

    auto depthStencilAttachment = ([&] () -> std::optional<WebCore::WebGPU::RenderPassDepthStencilAttachment> {
        if (renderPassDescriptor.depthStencilAttachment)
            return convertFromBacking(*renderPassDescriptor.depthStencilAttachment);
        return std::nullopt;
    })();
    if (renderPassDescriptor.depthStencilAttachment && !depthStencilAttachment)
        return std::nullopt;

    WeakPtr<WebCore::WebGPU::QuerySet> occlusionQuerySet;
    if (renderPassDescriptor.occlusionQuerySet) {
        occlusionQuerySet = convertQuerySetFromBacking(renderPassDescriptor.occlusionQuerySet.value());
        if (!occlusionQuerySet)
            return std::nullopt;
    }

    auto timestampWrites = renderPassDescriptor.timestampWrites ? convertFromBacking(*renderPassDescriptor.timestampWrites) : std::nullopt;

    return { { WTFMove(*base), WTFMove(colorAttachments), WTFMove(depthStencilAttachment), occlusionQuerySet, WTFMove(timestampWrites), renderPassDescriptor.maxDrawCount } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
