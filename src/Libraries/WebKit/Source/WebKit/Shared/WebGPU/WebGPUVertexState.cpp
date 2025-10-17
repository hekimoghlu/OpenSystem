/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#include "WebGPUVertexState.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUVertexState.h>

namespace WebKit::WebGPU {

std::optional<VertexState> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::VertexState& vertexState)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ProgrammableStage&>(vertexState));
    if (!base)
        return std::nullopt;

    Vector<std::optional<VertexBufferLayout>> buffers;
    buffers.reserveInitialCapacity(vertexState.buffers.size());
    for (const auto& buffer : vertexState.buffers) {
        if (buffer) {
            auto convertedBuffer = convertToBacking(*buffer);
            if (!convertedBuffer)
                return std::nullopt;
            buffers.append(WTFMove(convertedBuffer));
        } else
            buffers.append(std::nullopt);
    }

    return { { WTFMove(*base), WTFMove(buffers) } };
}

std::optional<WebCore::WebGPU::VertexState> ConvertFromBackingContext::convertFromBacking(const VertexState& vertexState)
{
    auto base = convertFromBacking(static_cast<const ProgrammableStage&>(vertexState));
    if (!base)
        return std::nullopt;

    Vector<std::optional<WebCore::WebGPU::VertexBufferLayout>> buffers;
    buffers.reserveInitialCapacity(vertexState.buffers.size());
    for (const auto& backingBuffer : vertexState.buffers) {
        if (backingBuffer) {
            auto buffer = convertFromBacking(*backingBuffer);
            if (!buffer)
                return std::nullopt;
            buffers.append(WTFMove(*buffer));
        } else
            buffers.append(std::nullopt);
    }

    return { { WTFMove(*base), WTFMove(buffers) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
