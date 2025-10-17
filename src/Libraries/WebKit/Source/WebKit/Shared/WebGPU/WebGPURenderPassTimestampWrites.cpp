/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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
#include "WebGPURenderPassTimestampWrites.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUQuerySet.h>
#include <WebCore/WebGPURenderPassTimestampWrites.h>

namespace WebKit::WebGPU {

std::optional<RenderPassTimestampWrites> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderPassTimestampWrites& renderPassTimestampWrite)
{
    if (!renderPassTimestampWrite.querySet)
        return std::nullopt;

    auto querySet = convertToBacking(*renderPassTimestampWrite.protectedQuerySet());

    return { { querySet, renderPassTimestampWrite.beginningOfPassWriteIndex, renderPassTimestampWrite.endOfPassWriteIndex } };
}

std::optional<WebCore::WebGPU::RenderPassTimestampWrites> ConvertFromBackingContext::convertFromBacking(const RenderPassTimestampWrites& renderPassTimestampWrite)
{
    WeakPtr querySet = convertQuerySetFromBacking(renderPassTimestampWrite.querySet);
    if (!querySet)
        return std::nullopt;

    return { { querySet, renderPassTimestampWrite.beginningOfPassWriteIndex, renderPassTimestampWrite.endOfPassWriteIndex } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
