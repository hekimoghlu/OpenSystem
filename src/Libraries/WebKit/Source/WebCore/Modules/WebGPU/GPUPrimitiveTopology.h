/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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

#include "WebGPUPrimitiveTopology.h"
#include <cstdint>

namespace WebCore {

enum class GPUPrimitiveTopology : uint8_t {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
};

inline WebGPU::PrimitiveTopology convertToBacking(GPUPrimitiveTopology primitiveTopology)
{
    switch (primitiveTopology) {
    case GPUPrimitiveTopology::PointList:
        return WebGPU::PrimitiveTopology::PointList;
    case GPUPrimitiveTopology::LineList:
        return WebGPU::PrimitiveTopology::LineList;
    case GPUPrimitiveTopology::LineStrip:
        return WebGPU::PrimitiveTopology::LineStrip;
    case GPUPrimitiveTopology::TriangleList:
        return WebGPU::PrimitiveTopology::TriangleList;
    case GPUPrimitiveTopology::TriangleStrip:
        return WebGPU::PrimitiveTopology::TriangleStrip;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
