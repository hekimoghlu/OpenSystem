/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 12, 2024.
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
#include "WebGPUBindGroupLayoutDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUBindGroupLayoutDescriptor.h>

namespace WebKit::WebGPU {

std::optional<BindGroupLayoutDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::BindGroupLayoutDescriptor& bindGroupLayoutDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ObjectDescriptorBase&>(bindGroupLayoutDescriptor));
    if (!base)
        return std::nullopt;

    Vector<BindGroupLayoutEntry> entries;
    entries.reserveInitialCapacity(bindGroupLayoutDescriptor.entries.size());
    for (const auto& entry : bindGroupLayoutDescriptor.entries) {
        auto convertedEntry = convertToBacking(entry);
        if (!convertedEntry)
            return std::nullopt;
        entries.append(WTFMove(*convertedEntry));
    }

    return { { WTFMove(*base), WTFMove(entries) } };
}

std::optional<WebCore::WebGPU::BindGroupLayoutDescriptor> ConvertFromBackingContext::convertFromBacking(const BindGroupLayoutDescriptor& bindGroupLayoutDescriptor)
{
    auto base = convertFromBacking(static_cast<const ObjectDescriptorBase&>(bindGroupLayoutDescriptor));
    if (!base)
        return std::nullopt;

    Vector<WebCore::WebGPU::BindGroupLayoutEntry> entries;
    entries.reserveInitialCapacity(bindGroupLayoutDescriptor.entries.size());
    for (const auto& backingEntry : bindGroupLayoutDescriptor.entries) {
        auto entry = convertFromBacking(backingEntry);
        if (!entry)
            return std::nullopt;
        entries.append(WTFMove(*entry));
    }

    return { { WTFMove(*base), WTFMove(entries) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
