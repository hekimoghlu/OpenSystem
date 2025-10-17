/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include "WebGPUStencilFaceState.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUStencilFaceState.h>

namespace WebKit::WebGPU {

std::optional<StencilFaceState> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::StencilFaceState& stencilFaceState)
{
    return { { stencilFaceState.compare, stencilFaceState.failOp, stencilFaceState.depthFailOp, stencilFaceState.passOp } };
}

std::optional<WebCore::WebGPU::StencilFaceState> ConvertFromBackingContext::convertFromBacking(const StencilFaceState& stencilFaceState)
{
    return { { stencilFaceState.compare, stencilFaceState.failOp, stencilFaceState.depthFailOp, stencilFaceState.passOp } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
