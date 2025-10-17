/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 27, 2023.
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
#include "WebGPUTextureDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUTextureDescriptor.h>

namespace WebKit::WebGPU {

std::optional<TextureDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::TextureDescriptor& textureDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ObjectDescriptorBase&>(textureDescriptor));
    if (!base)
        return std::nullopt;

    auto size = convertToBacking(textureDescriptor.size);
    if (!size)
        return std::nullopt;

    return { { WTFMove(*base), WTFMove(*size), textureDescriptor.mipLevelCount, textureDescriptor.sampleCount, textureDescriptor.dimension, textureDescriptor.format, textureDescriptor.usage, textureDescriptor.viewFormats } };
}

std::optional<WebCore::WebGPU::TextureDescriptor> ConvertFromBackingContext::convertFromBacking(const TextureDescriptor& textureDescriptor)
{
    auto base = convertFromBacking(static_cast<const ObjectDescriptorBase&>(textureDescriptor));
    if (!base)
        return std::nullopt;

    auto size = convertFromBacking(textureDescriptor.size);
    if (!size)
        return std::nullopt;

    return { { WTFMove(*base), WTFMove(*size), textureDescriptor.mipLevelCount, textureDescriptor.sampleCount, textureDescriptor.dimension, textureDescriptor.format, textureDescriptor.usage, textureDescriptor.viewFormats } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
