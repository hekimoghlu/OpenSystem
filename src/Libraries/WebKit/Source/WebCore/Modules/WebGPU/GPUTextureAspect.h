/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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

#include "WebGPUTextureAspect.h"
#include <cstdint>

namespace WebCore {

enum class GPUTextureAspect : uint8_t {
    All,
    StencilOnly,
    DepthOnly,
};

inline WebGPU::TextureAspect convertToBacking(GPUTextureAspect textureAspect)
{
    switch (textureAspect) {
    case GPUTextureAspect::All:
        return WebGPU::TextureAspect::All;
    case GPUTextureAspect::StencilOnly:
        return WebGPU::TextureAspect::StencilOnly;
    case GPUTextureAspect::DepthOnly:
        return WebGPU::TextureAspect::DepthOnly;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
