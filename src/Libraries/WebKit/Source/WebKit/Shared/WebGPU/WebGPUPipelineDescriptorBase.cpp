/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#include "WebGPUPipelineDescriptorBase.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUPipelineDescriptorBase.h>
#include <WebCore/WebGPUPipelineLayout.h>

namespace WebKit::WebGPU {

std::optional<PipelineDescriptorBase> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::PipelineDescriptorBase& pipelineDescriptorBase)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ObjectDescriptorBase&>(pipelineDescriptorBase));
    if (!base)
        return std::nullopt;

    std::optional<WebGPUIdentifier> layout;
    if (pipelineDescriptorBase.layout) {
        layout = convertToBacking(*pipelineDescriptorBase.protectedLayout().get());
        if (!layout)
            return std::nullopt;
    }

    return { { WTFMove(*base), layout } };
}

std::optional<WebCore::WebGPU::PipelineDescriptorBase> ConvertFromBackingContext::convertFromBacking(const PipelineDescriptorBase& pipelineDescriptorBase)
{
    auto base = convertFromBacking(static_cast<const ObjectDescriptorBase&>(pipelineDescriptorBase));
    if (!base)
        return std::nullopt;

    if (!pipelineDescriptorBase.layout)
        return std::nullopt;

    auto layout = convertPipelineLayoutFromBacking(*pipelineDescriptorBase.layout);
    if (!layout)
        return std::nullopt;

    return { { WTFMove(*base), layout } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
