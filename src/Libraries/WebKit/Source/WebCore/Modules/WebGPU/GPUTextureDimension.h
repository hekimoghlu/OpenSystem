/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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

#include "WebGPUTextureDimension.h"
#include <cstdint>

namespace WebCore {

enum class GPUTextureDimension : uint8_t {
    _1d,
    _2d,
    _3d,
};

inline WebGPU::TextureDimension convertToBacking(GPUTextureDimension textureDimension)
{
    switch (textureDimension) {
    case GPUTextureDimension::_1d:
        return WebGPU::TextureDimension::_1d;
    case GPUTextureDimension::_2d:
        return WebGPU::TextureDimension::_2d;
    case GPUTextureDimension::_3d:
        return WebGPU::TextureDimension::_3d;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

}
