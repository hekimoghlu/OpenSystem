/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
#include "WebGPUSamplerDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUSamplerDescriptor.h>

namespace WebKit::WebGPU {

std::optional<SamplerDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::SamplerDescriptor& samplerDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ObjectDescriptorBase&>(samplerDescriptor));
    if (!base)
        return std::nullopt;

    return { { WTFMove(*base), samplerDescriptor.addressModeU, samplerDescriptor.addressModeV, samplerDescriptor.addressModeW, samplerDescriptor.magFilter, samplerDescriptor.minFilter, samplerDescriptor.mipmapFilter, samplerDescriptor.lodMinClamp, samplerDescriptor.lodMaxClamp, samplerDescriptor.compare, samplerDescriptor.maxAnisotropy } };
}

std::optional<WebCore::WebGPU::SamplerDescriptor> ConvertFromBackingContext::convertFromBacking(const SamplerDescriptor& samplerDescriptor)
{
    auto base = convertFromBacking(static_cast<const ObjectDescriptorBase&>(samplerDescriptor));
    if (!base)
        return std::nullopt;

    return { { WTFMove(*base), samplerDescriptor.addressModeU, samplerDescriptor.addressModeV, samplerDescriptor.addressModeW, samplerDescriptor.magFilter, samplerDescriptor.minFilter, samplerDescriptor.mipmapFilter, samplerDescriptor.lodMinClamp, samplerDescriptor.lodMaxClamp, samplerDescriptor.compare, samplerDescriptor.maxAnisotropy } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
