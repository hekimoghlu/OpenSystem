/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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

#include "WebGPUSamplerBindingType.h"
#include <cstdint>

namespace WebCore {

enum class GPUSamplerBindingType : uint8_t {
    Filtering,
    NonFiltering,
    Comparison,
};

inline WebGPU::SamplerBindingType convertToBacking(GPUSamplerBindingType samplerBindingType)
{
    switch (samplerBindingType) {
    case GPUSamplerBindingType::Filtering:
        return WebGPU::SamplerBindingType::Filtering;
    case GPUSamplerBindingType::NonFiltering:
        return WebGPU::SamplerBindingType::NonFiltering;
    case GPUSamplerBindingType::Comparison:
        return WebGPU::SamplerBindingType::Comparison;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
