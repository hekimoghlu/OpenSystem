/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
#include "WebGPUFragmentState.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUFragmentState.h>

namespace WebKit::WebGPU {

std::optional<FragmentState> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::FragmentState& fragmentState)
{
    auto base = convertToBacking(static_cast<const WebCore::WebGPU::ProgrammableStage&>(fragmentState));
    if (!base)
        return std::nullopt;

    Vector<std::optional<ColorTargetState>> targets;
    targets.reserveInitialCapacity(fragmentState.targets.size());
    for (const auto& target : fragmentState.targets) {
        if (target) {
            auto convertedTarget = convertToBacking(*target);
            if (!convertedTarget)
                return std::nullopt;
            targets.append(WTFMove(*convertedTarget));
        } else
            targets.append(std::nullopt);
    }

    return { { WTFMove(*base), WTFMove(targets) } };
}

std::optional<WebCore::WebGPU::FragmentState> ConvertFromBackingContext::convertFromBacking(const FragmentState& fragmentState)
{
    auto base = convertFromBacking(static_cast<const ProgrammableStage&>(fragmentState));
    if (!base)
        return std::nullopt;

    Vector<std::optional<WebCore::WebGPU::ColorTargetState>> targets;
    targets.reserveInitialCapacity(fragmentState.targets.size());
    for (const auto& backingTarget : fragmentState.targets) {
        if (backingTarget) {
            auto target = convertFromBacking(*backingTarget);
            if (!target)
                return std::nullopt;
            targets.append(WTFMove(*target));
        } else
            targets.append(std::nullopt);
    }

    return { { WTFMove(*base), WTFMove(targets) } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
