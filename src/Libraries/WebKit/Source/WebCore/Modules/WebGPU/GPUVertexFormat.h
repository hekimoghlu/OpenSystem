/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 15, 2024.
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

#include "WebGPUVertexFormat.h"
#include <cstdint>

namespace WebCore {

enum class GPUVertexFormat : uint8_t {
    Uint8x2,
    Uint8x4,
    Sint8x2,
    Sint8x4,
    Unorm8x2,
    Unorm8x4,
    Snorm8x2,
    Snorm8x4,
    Uint16x2,
    Uint16x4,
    Sint16x2,
    Sint16x4,
    Unorm16x2,
    Unorm16x4,
    Snorm16x2,
    Snorm16x4,
    Float16x2,
    Float16x4,
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
    Unorm1010102,
};

inline WebGPU::VertexFormat convertToBacking(GPUVertexFormat vertexFormat)
{
    switch (vertexFormat) {
    case GPUVertexFormat::Uint8x2:
        return WebGPU::VertexFormat::Uint8x2;
    case GPUVertexFormat::Uint8x4:
        return WebGPU::VertexFormat::Uint8x4;
    case GPUVertexFormat::Sint8x2:
        return WebGPU::VertexFormat::Sint8x2;
    case GPUVertexFormat::Sint8x4:
        return WebGPU::VertexFormat::Sint8x4;
    case GPUVertexFormat::Unorm8x2:
        return WebGPU::VertexFormat::Unorm8x2;
    case GPUVertexFormat::Unorm8x4:
        return WebGPU::VertexFormat::Unorm8x4;
    case GPUVertexFormat::Snorm8x2:
        return WebGPU::VertexFormat::Snorm8x2;
    case GPUVertexFormat::Snorm8x4:
        return WebGPU::VertexFormat::Snorm8x4;
    case GPUVertexFormat::Uint16x2:
        return WebGPU::VertexFormat::Uint16x2;
    case GPUVertexFormat::Uint16x4:
        return WebGPU::VertexFormat::Uint16x4;
    case GPUVertexFormat::Sint16x2:
        return WebGPU::VertexFormat::Sint16x2;
    case GPUVertexFormat::Sint16x4:
        return WebGPU::VertexFormat::Sint16x4;
    case GPUVertexFormat::Unorm16x2:
        return WebGPU::VertexFormat::Unorm16x2;
    case GPUVertexFormat::Unorm16x4:
        return WebGPU::VertexFormat::Unorm16x4;
    case GPUVertexFormat::Snorm16x2:
        return WebGPU::VertexFormat::Snorm16x2;
    case GPUVertexFormat::Snorm16x4:
        return WebGPU::VertexFormat::Snorm16x4;
    case GPUVertexFormat::Float16x2:
        return WebGPU::VertexFormat::Float16x2;
    case GPUVertexFormat::Float16x4:
        return WebGPU::VertexFormat::Float16x4;
    case GPUVertexFormat::Float32:
        return WebGPU::VertexFormat::Float32;
    case GPUVertexFormat::Float32x2:
        return WebGPU::VertexFormat::Float32x2;
    case GPUVertexFormat::Float32x3:
        return WebGPU::VertexFormat::Float32x3;
    case GPUVertexFormat::Float32x4:
        return WebGPU::VertexFormat::Float32x4;
    case GPUVertexFormat::Uint32:
        return WebGPU::VertexFormat::Uint32;
    case GPUVertexFormat::Uint32x2:
        return WebGPU::VertexFormat::Uint32x2;
    case GPUVertexFormat::Uint32x3:
        return WebGPU::VertexFormat::Uint32x3;
    case GPUVertexFormat::Uint32x4:
        return WebGPU::VertexFormat::Uint32x4;
    case GPUVertexFormat::Sint32:
        return WebGPU::VertexFormat::Sint32;
    case GPUVertexFormat::Sint32x2:
        return WebGPU::VertexFormat::Sint32x2;
    case GPUVertexFormat::Sint32x3:
        return WebGPU::VertexFormat::Sint32x3;
    case GPUVertexFormat::Sint32x4:
        return WebGPU::VertexFormat::Sint32x4;
    case GPUVertexFormat::Unorm1010102:
        return WebGPU::VertexFormat::Unorm10_10_10_2;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
