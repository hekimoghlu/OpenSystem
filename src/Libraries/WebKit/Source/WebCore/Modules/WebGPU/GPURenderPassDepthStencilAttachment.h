/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 15, 2021.
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
#include "GPULoadOp.h"
#include "GPUStoreOp.h"
#include "GPUTextureView.h"
#include "WebGPURenderPassDepthStencilAttachment.h"
#include <variant>
#include <wtf/RefPtr.h>

namespace WebCore {

struct GPURenderPassDepthStencilAttachment {
    WebGPU::RenderPassDepthStencilAttachment convertToBacking() const
    {
        ASSERT(view);
        return {
            view->backing(),
            depthClearValue.value_or(-1.f),
            depthLoadOp ? std::optional { WebCore::convertToBacking(*depthLoadOp) } : std::nullopt,
            depthStoreOp ? std::optional { WebCore::convertToBacking(*depthStoreOp) } : std::nullopt,
            depthReadOnly,
            stencilClearValue,
            stencilLoadOp ? std::optional { WebCore::convertToBacking(*stencilLoadOp) } : std::nullopt,
            stencilStoreOp ? std::optional { WebCore::convertToBacking(*stencilStoreOp) } : std::nullopt,
            stencilReadOnly,
        };
    }

    WeakPtr<GPUTextureView> view;

    std::optional<float> depthClearValue;
    std::optional<GPULoadOp> depthLoadOp;
    std::optional<GPUStoreOp> depthStoreOp;
    bool depthReadOnly { false };

    GPUStencilValue stencilClearValue { 0 };
    std::optional<GPULoadOp> stencilLoadOp;
    std::optional<GPUStoreOp> stencilStoreOp;
    bool stencilReadOnly { false };
};

}
