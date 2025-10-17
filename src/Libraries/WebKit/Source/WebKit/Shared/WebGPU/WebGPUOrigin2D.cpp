/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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
#include "WebGPUOrigin2D.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUOrigin2D.h>

namespace WebKit::WebGPU {

std::optional<Origin2DDict> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Origin2DDict& origin2DDict)
{
    return { { origin2DDict.x, origin2DDict.y } };
}

std::optional<Origin2D> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Origin2D& origin2D)
{
    return WTF::switchOn(origin2D, [] (const Vector<WebCore::WebGPU::IntegerCoordinate>& vector) -> std::optional<Origin2D> {
        return { { vector } };
    }, [this] (const WebCore::WebGPU::Origin2DDict& origin2DDict) -> std::optional<Origin2D> {
        auto origin2D = convertToBacking(origin2DDict);
        if (!origin2D)
            return std::nullopt;
        return { { WTFMove(*origin2D) } };
    });
}

std::optional<WebCore::WebGPU::Origin2DDict> ConvertFromBackingContext::convertFromBacking(const Origin2DDict& origin2DDict)
{
    return { { origin2DDict.x, origin2DDict.y } };
}

std::optional<WebCore::WebGPU::Origin2D> ConvertFromBackingContext::convertFromBacking(const Origin2D& origin2D)
{
    return WTF::switchOn(origin2D, [] (const Vector<WebCore::WebGPU::IntegerCoordinate>& vector) -> std::optional<WebCore::WebGPU::Origin2D> {
        return { { vector } };
    }, [this] (const Origin2DDict& origin2DDict) -> std::optional<WebCore::WebGPU::Origin2D> {
        auto origin2D = convertFromBacking(origin2DDict);
        if (!origin2D)
            return std::nullopt;
        return { { WTFMove(*origin2D) } };
    });
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
