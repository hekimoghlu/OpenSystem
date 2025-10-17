/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 19, 2022.
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

#include "GPUCullMode.h"
#include "GPUFrontFace.h"
#include "GPUIndexFormat.h"
#include "GPUIntegralTypes.h"
#include "GPUPrimitiveTopology.h"
#include "WebGPUPrimitiveState.h"
#include <optional>

namespace WebCore {

struct GPUPrimitiveState {
    WebGPU::PrimitiveState convertToBacking() const
    {
        return {
            WebCore::convertToBacking(topology),
            stripIndexFormat ? std::optional { WebCore::convertToBacking(*stripIndexFormat) } : std::nullopt,
            WebCore::convertToBacking(frontFace),
            WebCore::convertToBacking(cullMode),
            unclippedDepth,
        };
    }

    GPUPrimitiveTopology topology { GPUPrimitiveTopology::TriangleList };
    std::optional<GPUIndexFormat> stripIndexFormat;
    GPUFrontFace frontFace { GPUFrontFace::Ccw };
    GPUCullMode cullMode { GPUCullMode::None };

    // Requires "depth-clip-control" feature.
    bool unclippedDepth { false };
};

}
