/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#include "WebGPURenderPassColorAttachment.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPURenderPassColorAttachment.h>
#include <WebCore/WebGPUTextureView.h>

namespace WebKit::WebGPU {

std::optional<RenderPassColorAttachment> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderPassColorAttachment& renderPassColorAttachment)
{
    auto view = convertToBacking(renderPassColorAttachment.protectedView().get());

    std::optional<WebGPUIdentifier> resolveTarget;
    if (renderPassColorAttachment.resolveTarget) {
        resolveTarget = convertToBacking(*renderPassColorAttachment.protectedResolveTarget());
        if (!resolveTarget)
            return std::nullopt;
    }

    std::optional<Color> clearValue;
    if (renderPassColorAttachment.clearValue) {
        clearValue = convertToBacking(*renderPassColorAttachment.clearValue);
        if (!clearValue)
            return std::nullopt;
    }

    return { { view, renderPassColorAttachment.depthSlice, resolveTarget, WTFMove(clearValue), renderPassColorAttachment.loadOp, renderPassColorAttachment.storeOp } };
}

std::optional<WebCore::WebGPU::RenderPassColorAttachment> ConvertFromBackingContext::convertFromBacking(const RenderPassColorAttachment& renderPassColorAttachment)
{
    WeakPtr view = convertTextureViewFromBacking(renderPassColorAttachment.view);
    if (!view)
        return std::nullopt;

    WeakPtr<WebCore::WebGPU::TextureView> resolveTarget;
    if (renderPassColorAttachment.resolveTarget) {
        resolveTarget = convertTextureViewFromBacking(renderPassColorAttachment.resolveTarget.value());
        if (!resolveTarget)
            return std::nullopt;
    }

    std::optional<WebCore::WebGPU::Color> clearValue;
    if (renderPassColorAttachment.clearValue) {
        clearValue = convertFromBacking(*renderPassColorAttachment.clearValue);
        if (!clearValue)
            return std::nullopt;
    }

    return { { *view, renderPassColorAttachment.depthSlice, resolveTarget, WTFMove(clearValue), renderPassColorAttachment.loadOp, renderPassColorAttachment.storeOp } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
