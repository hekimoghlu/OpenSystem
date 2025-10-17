/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 9, 2023.
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
#include "WebGPUDepthStencilState.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUDepthStencilState.h>

namespace WebKit::WebGPU {

std::optional<DepthStencilState> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::DepthStencilState& depthStencilState)
{
    auto stencilFront = convertToBacking(depthStencilState.stencilFront);
    if (!stencilFront)
        return std::nullopt;

    auto stencilBack = convertToBacking(depthStencilState.stencilBack);
    if (!stencilBack)
        return std::nullopt;

    return { { depthStencilState.format, depthStencilState.depthWriteEnabled, depthStencilState.depthCompare, WTFMove(*stencilFront), WTFMove(*stencilBack), depthStencilState.stencilReadMask, depthStencilState.stencilWriteMask, depthStencilState.depthBias, depthStencilState.depthBiasSlopeScale, depthStencilState.depthBiasClamp } };
}

std::optional<WebCore::WebGPU::DepthStencilState> ConvertFromBackingContext::convertFromBacking(const DepthStencilState& depthStencilState)
{
    auto stencilFront = convertFromBacking(depthStencilState.stencilFront);
    if (!stencilFront)
        return std::nullopt;

    auto stencilBack = convertFromBacking(depthStencilState.stencilBack);
    if (!stencilBack)
        return std::nullopt;

    return { { depthStencilState.format, depthStencilState.depthWriteEnabled, depthStencilState.depthCompare, WTFMove(*stencilFront), WTFMove(*stencilBack), depthStencilState.stencilReadMask, depthStencilState.stencilWriteMask, depthStencilState.depthBias, depthStencilState.depthBiasSlopeScale, depthStencilState.depthBiasClamp } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
