/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#include "WebGPUExtent3D.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUExtent3D.h>

namespace WebKit::WebGPU {

std::optional<Extent3DDict> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Extent3DDict& extent3DDict)
{
    return { { extent3DDict.width, extent3DDict.height, extent3DDict.depthOrArrayLayers } };
}

std::optional<Extent3D> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Extent3D& extent3D)
{
    return WTF::switchOn(extent3D, [] (const Vector<WebCore::WebGPU::IntegerCoordinate>& vector) -> std::optional<Extent3D> {
        return { { vector } };
    }, [this] (const WebCore::WebGPU::Extent3DDict& extent3DDict) -> std::optional<Extent3D> {
        auto extent3D = convertToBacking(extent3DDict);
        if (!extent3D)
            return std::nullopt;
        return { { WTFMove(*extent3D) } };
    });
}

std::optional<WebCore::WebGPU::Extent3DDict> ConvertFromBackingContext::convertFromBacking(const Extent3DDict& extent3DDict)
{
    return { { extent3DDict.width, extent3DDict.height, extent3DDict.depthOrArrayLayers } };
}

std::optional<WebCore::WebGPU::Extent3D> ConvertFromBackingContext::convertFromBacking(const Extent3D& extent3D)
{
    return WTF::switchOn(extent3D, [] (const Vector<WebCore::WebGPU::IntegerCoordinate>& vector) -> std::optional<WebCore::WebGPU::Extent3D> {
        return { { vector } };
    }, [this] (const Extent3DDict& extent3DDict) -> std::optional<WebCore::WebGPU::Extent3D> {
        auto extent3D = convertFromBacking(extent3DDict);
        if (!extent3D)
            return std::nullopt;
        return { { WTFMove(*extent3D) } };
    });
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
