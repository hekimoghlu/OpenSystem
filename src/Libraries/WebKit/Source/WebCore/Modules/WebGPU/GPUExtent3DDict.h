/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#include "WebGPUExtent3D.h"
#include <variant>
#include <wtf/Vector.h>

namespace WebCore {

struct GPUExtent3DDict {
    WebGPU::Extent3DDict convertToBacking() const
    {
        return {
            width,
            height,
            depthOrArrayLayers,
        };
    }

    GPUIntegerCoordinate width { 0 };
    GPUIntegerCoordinate height { 0 };
    GPUIntegerCoordinate depthOrArrayLayers { 0 };
};

using GPUExtent3D = std::variant<Vector<GPUIntegerCoordinate>, GPUExtent3DDict>;

inline WebGPU::Extent3D convertToBacking(const GPUExtent3D& extent3D)
{
    return WTF::switchOn(extent3D, [](const Vector<GPUIntegerCoordinate>& vector) -> WebGPU::Extent3D {
        return vector;
    }, [](const GPUExtent3DDict& extent3D) -> WebGPU::Extent3D {
        return extent3D.convertToBacking();
    });
}

}
