/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#include "WebGPUShaderModuleDescriptor.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUShaderModuleDescriptor.h>

namespace WebKit::WebGPU {

std::optional<ShaderModuleDescriptor> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ShaderModuleDescriptor& shaderModuleDescriptor)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ObjectDescriptorBase&>(shaderModuleDescriptor));
    if (!base)
        return std::nullopt;

    Vector<KeyValuePair<String, ShaderModuleCompilationHint>> hints;
    hints.reserveInitialCapacity(shaderModuleDescriptor.hints.size());
    for (const auto& hint : shaderModuleDescriptor.hints) {
        auto value = convertToBacking(hint.value);
        if (!value)
            return std::nullopt;
        hints.append(makeKeyValuePair(hint.key, WTFMove(*value)));
    }

    return { { WTFMove(*base), shaderModuleDescriptor.code, WTFMove(hints) } };
}

std::optional<WebCore::WebGPU::ShaderModuleDescriptor> ConvertFromBackingContext::convertFromBacking(const ShaderModuleDescriptor& shaderModuleDescriptor)
{
    auto base = convertFromBacking(static_cast<const ObjectDescriptorBase&>(shaderModuleDescriptor));
    if (!base)
        return std::nullopt;

    Vector<KeyValuePair<String, WebCore::WebGPU::ShaderModuleCompilationHint>> hints;
    hints.reserveInitialCapacity(shaderModuleDescriptor.hints.size());
    for (const auto& hint : shaderModuleDescriptor.hints) {
        auto value = convertFromBacking(hint.value);
        if (!value)
            return std::nullopt;
        hints.append(makeKeyValuePair(hint.key, WTFMove(*value)));
    }

    return { { WTFMove(*base), shaderModuleDescriptor.code, WTFMove(hints) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
