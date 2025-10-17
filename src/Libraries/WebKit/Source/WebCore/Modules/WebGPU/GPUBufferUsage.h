/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#include "WebGPUBufferUsage.h"
#include <cstdint>
#include <wtf/RefCounted.h>

namespace WebCore {

using GPUBufferUsageFlags = uint32_t;

class GPUBufferUsage : public RefCounted<GPUBufferUsage> {
public:
    static constexpr GPUFlagsConstant MAP_READ      = 0x0001;
    static constexpr GPUFlagsConstant MAP_WRITE     = 0x0002;
    static constexpr GPUFlagsConstant COPY_SRC      = 0x0004;
    static constexpr GPUFlagsConstant COPY_DST      = 0x0008;
    static constexpr GPUFlagsConstant INDEX         = 0x0010;
    static constexpr GPUFlagsConstant VERTEX        = 0x0020;
    static constexpr GPUFlagsConstant UNIFORM       = 0x0040;
    static constexpr GPUFlagsConstant STORAGE       = 0x0080;
    static constexpr GPUFlagsConstant INDIRECT      = 0x0100;
    static constexpr GPUFlagsConstant QUERY_RESOLVE = 0x0200;
};

static constexpr bool compare(unsigned a, WebCore::WebGPU::BufferUsage b)
{
    return a == static_cast<unsigned>(b);
}

inline WebGPU::BufferUsageFlags convertBufferUsageFlagsToBacking(GPUBufferUsageFlags bufferUsageFlags)
{
    static_assert(compare(GPUBufferUsage::MAP_READ, WebGPU::BufferUsage::MapRead), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::MAP_WRITE,  WebGPU::BufferUsage::MapWrite), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::COPY_SRC,  WebGPU::BufferUsage::CopySource), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::COPY_DST,  WebGPU::BufferUsage::CopyDestination), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::INDEX, WebGPU::BufferUsage::Index), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::VERTEX, WebGPU::BufferUsage::Vertex), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::UNIFORM, WebGPU::BufferUsage::Uniform), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::STORAGE, WebGPU::BufferUsage::Storage), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::INDIRECT, WebGPU::BufferUsage::Indirect), "GPUBufferUsageFlags does not match BufferUsageFlags");
    static_assert(compare(GPUBufferUsage::QUERY_RESOLVE, WebGPU::BufferUsage::QueryResolve), "GPUBufferUsageFlags does not match BufferUsageFlags");
    return static_cast<WebGPU::BufferUsageFlags>(bufferUsageFlags);
}

}
