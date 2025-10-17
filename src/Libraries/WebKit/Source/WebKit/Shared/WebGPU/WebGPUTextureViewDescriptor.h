/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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

#include "WebGPUObjectDescriptorBase.h"
#include <WebCore/WebGPUIntegralTypes.h>
#include <WebCore/WebGPUTextureAspect.h>
#include <WebCore/WebGPUTextureFormat.h>
#include <WebCore/WebGPUTextureViewDimension.h>
#include <optional>

namespace WebKit::WebGPU {

struct TextureViewDescriptor : public ObjectDescriptorBase {
    std::optional<WebCore::WebGPU::TextureFormat> format;
    std::optional<WebCore::WebGPU::TextureViewDimension> dimension;
    WebCore::WebGPU::TextureAspect aspect { WebCore::WebGPU::TextureAspect::All };
    WebCore::WebGPU::IntegerCoordinate baseMipLevel { 0 };
    std::optional<WebCore::WebGPU::IntegerCoordinate> mipLevelCount;
    WebCore::WebGPU::IntegerCoordinate baseArrayLayer { 0 };
    std::optional<WebCore::WebGPU::IntegerCoordinate> arrayLayerCount;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
