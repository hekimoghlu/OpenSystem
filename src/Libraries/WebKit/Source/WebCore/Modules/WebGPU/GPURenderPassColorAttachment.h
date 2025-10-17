/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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

#include "GPUColorDict.h"
#include "GPUIntegralTypes.h"
#include "GPULoadOp.h"
#include "GPUStoreOp.h"
#include "GPUTextureView.h"
#include "WebGPURenderPassColorAttachment.h"
#include <variant>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

struct GPURenderPassColorAttachment {
    WebGPU::RenderPassColorAttachment convertToBacking() const
    {
        ASSERT(view);
        return {
            .view = view->backing(),
            .depthSlice = depthSlice,
            .resolveTarget = resolveTarget ? &resolveTarget->backing() : nullptr,
            .clearValue = clearValue ? std::optional { WebCore::convertToBacking(*clearValue) } : std::nullopt,
            .loadOp = WebCore::convertToBacking(loadOp),
            .storeOp = WebCore::convertToBacking(storeOp),
        };
    }

    WeakPtr<GPUTextureView> view;
    std::optional<GPUIntegerCoordinate> depthSlice;
    WeakPtr<GPUTextureView> resolveTarget;

    std::optional<GPUColor> clearValue;
    GPULoadOp loadOp { GPULoadOp::Load };
    GPUStoreOp storeOp { GPUStoreOp::Store };
};

}
