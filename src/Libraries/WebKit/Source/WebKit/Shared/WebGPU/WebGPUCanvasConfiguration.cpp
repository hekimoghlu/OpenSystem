/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "WebGPUCanvasConfiguration.h"

#if ENABLE(GPU_PROCESS)

#include "WebGPUConvertFromBackingContext.h"
#include "WebGPUConvertToBackingContext.h"
#include <WebCore/WebGPUCanvasConfiguration.h>
#include <WebCore/WebGPUDevice.h>

namespace WebKit::WebGPU {

std::optional<CanvasConfiguration> ConvertToBackingContext::convertToBacking(const WebCore::WebGPU::CanvasConfiguration& canvasConfiguration)
{
    Ref protectedDevice = canvasConfiguration.device.get();
    auto device = convertToBacking(protectedDevice.get());

    return { { device, canvasConfiguration.format, canvasConfiguration.usage, canvasConfiguration.viewFormats, canvasConfiguration.colorSpace, canvasConfiguration.toneMappingMode, canvasConfiguration.compositingAlphaMode, canvasConfiguration.reportValidationErrors } };
}

std::optional<WebCore::WebGPU::CanvasConfiguration> ConvertFromBackingContext::convertFromBacking(const CanvasConfiguration& canvasConfiguration)
{
    WeakPtr device = convertDeviceFromBacking(canvasConfiguration.device);
    if (!device)
        return std::nullopt;

    return { { *device, canvasConfiguration.format, canvasConfiguration.usage, canvasConfiguration.viewFormats, canvasConfiguration.colorSpace, canvasConfiguration.toneMappingMode, canvasConfiguration.compositingAlphaMode, canvasConfiguration.reportValidationErrors } };
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
