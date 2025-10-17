/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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

#include "WebGPUIntegralTypes.h"
#include "WebGPUObjectDescriptorBase.h"
#include "WebGPUTextureAspect.h"
#include "WebGPUTextureFormat.h"
#include "WebGPUTextureViewDimension.h"
#include <optional>

namespace WebCore::WebGPU {

struct TextureViewDescriptor : public ObjectDescriptorBase {
    std::optional<TextureFormat> format;
    std::optional<TextureViewDimension> dimension;
    TextureAspect aspect { TextureAspect::All };
    IntegerCoordinate baseMipLevel { 0 };
    std::optional<IntegerCoordinate> mipLevelCount;
    IntegerCoordinate baseArrayLayer { 0 };
    std::optional<IntegerCoordinate> arrayLayerCount;
};

} // namespace WebCore::WebGPU
