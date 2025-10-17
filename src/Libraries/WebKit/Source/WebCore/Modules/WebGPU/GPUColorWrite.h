/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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

#include "GPUBlendFactor.h"
#include "GPUBlendOperation.h"
#include "GPUIntegralTypes.h"
#include "WebGPUColorWrite.h"
#include <cstdint>
#include <wtf/RefCounted.h>

namespace WebCore {

using GPUColorWriteFlags = uint32_t;
class GPUColorWrite : public RefCounted<GPUColorWrite> {
public:
    static constexpr GPUFlagsConstant RED   = 0x1;
    static constexpr GPUFlagsConstant GREEN = 0x2;
    static constexpr GPUFlagsConstant BLUE  = 0x4;
    static constexpr GPUFlagsConstant ALPHA = 0x8;
    static constexpr GPUFlagsConstant ALL   = 0xF;
};

static constexpr bool compare(auto a, auto b)
{
    return static_cast<unsigned>(a) == static_cast<unsigned>(b);
}

inline WebGPU::ColorWriteFlags convertColorWriteFlagsToBacking(GPUColorWriteFlags colorWriteFlags)
{
    static_assert(compare(GPUColorWrite::RED, WebGPU::ColorWrite::Red), "ColorWriteFlags enum values differ");
    static_assert(compare(GPUColorWrite::GREEN, WebGPU::ColorWrite::Green), "ColorWriteFlags enum values differ");
    static_assert(compare(GPUColorWrite::BLUE, WebGPU::ColorWrite::Blue), "ColorWriteFlags enum values differ");
    static_assert(compare(GPUColorWrite::ALPHA, WebGPU::ColorWrite::Alpha), "ColorWriteFlags enum values differ");

    return static_cast<WebGPU::ColorWriteFlags>(colorWriteFlags);
}

}
