/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 13, 2023.
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

#if ENABLE(GPU_PROCESS)

#include "WebGPUStencilFaceState.h"
#include <WebCore/WebGPUCompareFunction.h>
#include <WebCore/WebGPUIntegralTypes.h>
#include <WebCore/WebGPUTextureFormat.h>
#include <optional>

namespace WebKit::WebGPU {

struct DepthStencilState {
    WebCore::WebGPU::TextureFormat format { WebCore::WebGPU::TextureFormat::R8unorm };

    std::optional<bool> depthWriteEnabled;
    std::optional<WebCore::WebGPU::CompareFunction> depthCompare;

    StencilFaceState stencilFront;
    StencilFaceState stencilBack;

    std::optional<WebCore::WebGPU::StencilValue> stencilReadMask;
    std::optional<WebCore::WebGPU::StencilValue> stencilWriteMask;

    WebCore::WebGPU::DepthBias depthBias { 0 };
    float depthBiasSlopeScale { 0 };
    float depthBiasClamp { 0 };
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
