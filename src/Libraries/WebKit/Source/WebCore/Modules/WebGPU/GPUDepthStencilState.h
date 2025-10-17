/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#include "GPUCompareFunction.h"
#include "GPUIntegralTypes.h"
#include "GPUStencilFaceState.h"
#include "GPUTextureFormat.h"
#include "WebGPUDepthStencilState.h"
#include <optional>

namespace WebCore {

struct GPUDepthStencilState {
    WebGPU::DepthStencilState convertToBacking() const
    {
        std::optional<WebGPU::CompareFunction> optionalDepthCompare;
        if (depthCompare)
            optionalDepthCompare = WebCore::convertToBacking(*depthCompare);
        return {
            .format = WebCore::convertToBacking(format),
            .depthWriteEnabled = depthWriteEnabled,
            .depthCompare = optionalDepthCompare,
            .stencilFront = stencilFront.convertToBacking(),
            .stencilBack = stencilBack.convertToBacking(),
            .stencilReadMask = stencilReadMask ? std::optional { *stencilReadMask } : std::nullopt,
            .stencilWriteMask = stencilWriteMask ? std::optional { *stencilWriteMask } : std::nullopt,
            .depthBias = depthBias,
            .depthBiasSlopeScale = depthBiasSlopeScale,
            .depthBiasClamp = depthBiasClamp,
        };
    }

    GPUTextureFormat format { GPUTextureFormat::R8unorm };

    std::optional<bool> depthWriteEnabled;
    std::optional<GPUCompareFunction> depthCompare;

    GPUStencilFaceState stencilFront;
    GPUStencilFaceState stencilBack;

    std::optional<GPUStencilValue> stencilReadMask;
    std::optional<GPUStencilValue> stencilWriteMask;

    GPUDepthBias depthBias { 0 };
    float depthBiasSlopeScale { 0 };
    float depthBiasClamp { 0 };
};

}
