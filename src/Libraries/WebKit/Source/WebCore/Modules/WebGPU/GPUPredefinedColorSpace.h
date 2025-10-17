/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 30, 2023.
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

#include "WebGPUPredefinedColorSpace.h"
#include <cstdint>

namespace WebCore {

enum class GPUPredefinedColorSpace : uint8_t {
    SRGB,
    DisplayP3
};

inline WebGPU::PredefinedColorSpace convertToBacking(GPUPredefinedColorSpace predefinedColorSpace)
{
    switch (predefinedColorSpace) {
    case GPUPredefinedColorSpace::SRGB:
        return WebGPU::PredefinedColorSpace::SRGB;
    case GPUPredefinedColorSpace::DisplayP3:
#if ENABLE(PREDEFINED_COLOR_SPACE_DISPLAY_P3)
        return WebGPU::PredefinedColorSpace::DisplayP3;
#else
        return WebGPU::PredefinedColorSpace::SRGB;
#endif
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
