/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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

#include "GPUObjectDescriptorBase.h"
#include "GPUQuerySet.h"
#include "GPURenderPassColorAttachment.h"
#include "GPURenderPassDepthStencilAttachment.h"
#include "GPURenderPassTimestampWrites.h"
#include "WebGPURenderPassDescriptor.h"
#include <optional>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

struct GPURenderPassDescriptor : public GPUObjectDescriptorBase {
    WebGPU::RenderPassDescriptor convertToBacking() const
    {
        return {
            { label },
            colorAttachments.map([](auto& colorAttachment) -> std::optional<WebGPU::RenderPassColorAttachment> {
                if (colorAttachment)
                    return colorAttachment->convertToBacking();
                return std::nullopt;
            }),
            depthStencilAttachment ? std::optional { depthStencilAttachment->convertToBacking() } : std::nullopt,
            occlusionQuerySet ? &occlusionQuerySet->backing() : nullptr,
            timestampWrites.convertToBacking(),
            maxDrawCount,
        };
    }

    Vector<std::optional<GPURenderPassColorAttachment>> colorAttachments;
    std::optional<GPURenderPassDepthStencilAttachment> depthStencilAttachment;
    WeakPtr<GPUQuerySet> occlusionQuerySet;
    GPURenderPassTimestampWrites timestampWrites;
    std::optional<GPUSize64> maxDrawCount;
};

}
