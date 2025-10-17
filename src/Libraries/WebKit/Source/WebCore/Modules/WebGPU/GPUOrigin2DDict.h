/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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
#include "WebGPUOrigin2D.h"
#include <variant>
#include <wtf/Vector.h>

namespace WebCore {

struct GPUOrigin2DDict {
    WebGPU::Origin2DDict convertToBacking() const
    {
        return {
            x,
            y,
        };
    }

    GPUIntegerCoordinate x { 0 };
    GPUIntegerCoordinate y { 0 };
};

using GPUOrigin2D = std::variant<Vector<GPUIntegerCoordinate>, GPUOrigin2DDict>;

inline WebGPU::Origin2D convertToBacking(const GPUOrigin2D& origin2D)
{
    return WTF::switchOn(origin2D, [](const Vector<GPUIntegerCoordinate>& vector) -> WebGPU::Origin2D {
        return vector;
    }, [](const GPUOrigin2DDict& origin2D) -> WebGPU::Origin2D {
        return origin2D.convertToBacking();
    });
}

}
