/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 20, 2023.
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
#include "WebGPUOrigin3D.h"
#include <variant>
#include <wtf/Vector.h>

namespace WebCore {

struct GPUOrigin3DDict {
    WebGPU::Origin3DDict convertToBacking() const
    {
        return {
            x,
            y,
            z,
        };
    }

    GPUIntegerCoordinate x { 0 };
    GPUIntegerCoordinate y { 0 };
    GPUIntegerCoordinate z { 0 };
};

using GPUOrigin3D = std::variant<Vector<GPUIntegerCoordinate>, GPUOrigin3DDict>;

inline WebGPU::Origin3D convertToBacking(const GPUOrigin3D& origin3D)
{
    return WTF::switchOn(origin3D, [](const Vector<GPUIntegerCoordinate>& vector) -> WebGPU::Origin3D {
        return vector;
    }, [](const GPUOrigin3DDict& origin3D) -> WebGPU::Origin3D {
        return origin3D.convertToBacking();
    });
}

}
