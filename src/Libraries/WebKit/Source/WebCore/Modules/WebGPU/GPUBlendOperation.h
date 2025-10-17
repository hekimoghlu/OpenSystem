/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 24, 2021.
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

#include "WebGPUBlendOperation.h"
#include <cstdint>

namespace WebCore {

enum class GPUBlendOperation : uint8_t {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
};

inline WebGPU::BlendOperation convertToBacking(GPUBlendOperation blendOperation)
{
    switch (blendOperation) {
    case GPUBlendOperation::Add:
        return WebGPU::BlendOperation::Add;
    case GPUBlendOperation::Subtract:
        return WebGPU::BlendOperation::Subtract;
    case GPUBlendOperation::ReverseSubtract:
        return WebGPU::BlendOperation::ReverseSubtract;
    case GPUBlendOperation::Min:
        return WebGPU::BlendOperation::Min;
    case GPUBlendOperation::Max:
        return WebGPU::BlendOperation::Max;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
