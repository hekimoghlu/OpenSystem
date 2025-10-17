/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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

#include "WebGPUCompareFunction.h"
#include <cstdint>

namespace WebCore {

enum class GPUCompareFunction : uint8_t {
    Never,
    Less,
    Equal,
    LessEqual,
    Greater,
    NotEqual,
    GreaterEqual,
    Always,
};

inline WebGPU::CompareFunction convertToBacking(GPUCompareFunction compareFunction)
{
    switch (compareFunction) {
    case GPUCompareFunction::Never:
        return WebGPU::CompareFunction::Never;
    case GPUCompareFunction::Less:
        return WebGPU::CompareFunction::Less;
    case GPUCompareFunction::Equal:
        return WebGPU::CompareFunction::Equal;
    case GPUCompareFunction::LessEqual:
        return WebGPU::CompareFunction::LessEqual;
    case GPUCompareFunction::Greater:
        return WebGPU::CompareFunction::Greater;
    case GPUCompareFunction::NotEqual:
        return WebGPU::CompareFunction::NotEqual;
    case GPUCompareFunction::GreaterEqual:
        return WebGPU::CompareFunction::GreaterEqual;
    case GPUCompareFunction::Always:
        return WebGPU::CompareFunction::Always;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
