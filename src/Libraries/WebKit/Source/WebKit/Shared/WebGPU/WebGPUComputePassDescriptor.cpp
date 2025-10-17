/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "WebGPUComputePassDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUComputePassDescriptor.h>

namespace WebKit::WebGPU {

std::optional<ComputePassDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ComputePassDescriptor& computePassDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ObjectDescriptorBase&>(computePassDescriptor));
    if (!base)
        return std::nullopt;

    auto timestampWrites = computePassDescriptor.timestampWrites ? convertToBacking(*computePassDescriptor.timestampWrites) : std::nullopt;

    return { { WTFMove(*base), timestampWrites } };
}

std::optional<WebCore::WebGPU::ComputePassDescriptor> ConvertFromBackingContext::convertFromBacking(const ComputePassDescriptor& computePassDescriptor)
{
    auto base = convertFromBacking(static_cast<const ObjectDescriptorBase&>(computePassDescriptor));
    if (!base)
        return std::nullopt;

    auto timestampWrites = computePassDescriptor.timestampWrites ? convertFromBacking(*computePassDescriptor.timestampWrites) : std::nullopt;

    return { { WTFMove(*base), timestampWrites } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
