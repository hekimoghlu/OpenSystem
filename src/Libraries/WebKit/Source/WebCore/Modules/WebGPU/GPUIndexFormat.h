/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#include "WebGPUIndexFormat.h"
#include <cstdint>

namespace WebCore {

enum class GPUIndexFormat : uint8_t {
    Uint16,
    Uint32,
};

inline WebGPU::IndexFormat convertToBacking(GPUIndexFormat indexFormat)
{
    switch (indexFormat) {
    case GPUIndexFormat::Uint16:
        return WebGPU::IndexFormat::Uint16;
    case GPUIndexFormat::Uint32:
        return WebGPU::IndexFormat::Uint32;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
