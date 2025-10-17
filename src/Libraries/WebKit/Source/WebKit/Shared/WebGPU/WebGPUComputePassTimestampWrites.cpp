/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 29, 2022.
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
#include "WebGPUComputePassTimestampWrites.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUComputePassTimestampWrites.h>
#include <WebCore/WebGPUDevice.h>
#include <WebCore/WebGPUQuerySet.h>

namespace WebKit::WebGPU {

std::optional<ComputePassTimestampWrites> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ComputePassTimestampWrites& computePassTimestampWrite)
{
    if (!computePassTimestampWrite.querySet)
        return std::nullopt;

    auto querySet = convertToBacking(*computePassTimestampWrite.protectedQuerySet());

    return { { querySet, computePassTimestampWrite.beginningOfPassWriteIndex, computePassTimestampWrite.endOfPassWriteIndex } };
}

std::optional<WebCore::WebGPU::ComputePassTimestampWrites> ConvertFromBackingContext::convertFromBacking(const ComputePassTimestampWrites& computePassTimestampWrites)
{
    WeakPtr querySet = convertQuerySetFromBacking(computePassTimestampWrites.querySet);
    if (!querySet)
        return std::nullopt;

    return { { querySet, computePassTimestampWrites.beginningOfPassWriteIndex, computePassTimestampWrites.endOfPassWriteIndex } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
