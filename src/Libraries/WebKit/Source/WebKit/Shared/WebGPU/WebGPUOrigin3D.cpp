/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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
#include "config.h"
#include "WebGPUOrigin3D.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUOrigin3D.h>

namespace WebKit::WebGPU {

std::optional<Origin3DDict> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Origin3DDict& origin3DDict)
{
    return { { origin3DDict.x, origin3DDict.y, origin3DDict.z } };
}

std::optional<Origin3D> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Origin3D& origin3D)
{
    return WTF::switchOn(origin3D, [] (const Vector<WebCore::WebGPU::IntegerCoordinate>& vector) -> std::optional<Origin3D> {
        return { { vector } };
    }, [this] (const WebCore::WebGPU::Origin3DDict& origin3DDict) -> std::optional<Origin3D> {
        auto origin3D = convertToBacking(origin3DDict);
        if (!origin3D)
            return std::nullopt;
        return { { WTFMove(*origin3D) } };
    });
}

std::optional<WebCore::WebGPU::Origin3DDict> ConvertFromBackingContext::convertFromBacking(const Origin3DDict& origin3DDict)
{
    return { { origin3DDict.x, origin3DDict.y, origin3DDict.z } };
}

std::optional<WebCore::WebGPU::Origin3D> ConvertFromBackingContext::convertFromBacking(const Origin3D& origin3D)
{
    return WTF::switchOn(origin3D, [] (const Vector<WebCore::WebGPU::IntegerCoordinate>& vector) -> std::optional<WebCore::WebGPU::Origin3D> {
        return { { vector } };
    }, [this] (const Origin3DDict& origin3DDict) -> std::optional<WebCore::WebGPU::Origin3D> {
        auto origin3D = convertFromBacking(origin3DDict);
        if (!origin3D)
            return std::nullopt;
        return { { WTFMove(*origin3D) } };
    });
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
