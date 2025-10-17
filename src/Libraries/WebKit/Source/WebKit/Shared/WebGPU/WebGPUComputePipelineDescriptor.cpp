/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 6, 2022.
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
#include "WebGPUComputePipelineDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUComputePipelineDescriptor.h>

namespace WebKit::WebGPU {

std::optional<ComputePipelineDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ComputePipelineDescriptor& computePipelineDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::PipelineDescriptorBase&>(computePipelineDescriptor));
    if (!base)
        return std::nullopt;

    auto compute = convertToBacking(computePipelineDescriptor.compute);
    if (!compute)
        return std::nullopt;

    return { { WTFMove(*base), WTFMove(*compute) } };
}

std::optional<WebCore::WebGPU::ComputePipelineDescriptor> ConvertFromBackingContext::convertFromBacking(const ComputePipelineDescriptor& computePipelineDescriptor)
{
    auto base = convertFromBacking(static_cast<const PipelineDescriptorBase&>(computePipelineDescriptor));
    if (!base)
        return std::nullopt;

    auto compute = convertFromBacking(computePipelineDescriptor.compute);
    if (!compute)
        return std::nullopt;

    return { { WTFMove(*base), WTFMove(*compute) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
