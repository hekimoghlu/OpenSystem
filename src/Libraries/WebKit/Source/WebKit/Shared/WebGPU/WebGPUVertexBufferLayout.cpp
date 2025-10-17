/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 27, 2023.
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
#include "WebGPUVertexBufferLayout.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUVertexBufferLayout.h>

namespace WebKit::WebGPU {

std::optional<VertexBufferLayout> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::VertexBufferLayout& vertexBufferLayout)
{
    Vector<VertexAttribute> attributes;
    attributes.reserveInitialCapacity(vertexBufferLayout.attributes.size());
    for (const auto& attribute : vertexBufferLayout.attributes) {
        auto convertedAttribute = convertToBacking(attribute);
        if (!convertedAttribute)
            return std::nullopt;
        attributes.append(WTFMove(*convertedAttribute));
    }

    return { { vertexBufferLayout.arrayStride, vertexBufferLayout.stepMode, WTFMove(attributes) } };
}

std::optional<WebCore::WebGPU::VertexBufferLayout> ConvertFromBackingContext::convertFromBacking(const VertexBufferLayout& vertexBufferLayout)
{
    Vector<WebCore::WebGPU::VertexAttribute> attributes;
    attributes.reserveInitialCapacity(vertexBufferLayout.attributes.size());
    for (const auto& backingAttribute : vertexBufferLayout.attributes) {
        auto attribute = convertFromBacking(backingAttribute);
        if (!attribute)
            return std::nullopt;
        attributes.append(WTFMove(*attribute));
    }

    return { { vertexBufferLayout.arrayStride, vertexBufferLayout.stepMode, WTFMove(attributes) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
