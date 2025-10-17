/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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
#pragma once

#include "GPUCanvasAlphaMode.h"
#include "GPUCanvasToneMapping.h"
#include "GPUDevice.h"
#include "GPUPredefinedColorSpace.h"
#include "GPUTextureFormat.h"
#include "GPUTextureUsage.h"
#include "WebGPUCanvasConfiguration.h"
#include <wtf/Vector.h>

namespace WebCore {

struct GPUCanvasConfiguration {
    WebGPU::CanvasConfiguration convertToBacking(bool reportValidationErrors) const
    {
        ASSERT(device);
        return {
            device->backing(),
            WebCore::convertToBacking(format),
            convertTextureUsageFlagsToBacking(usage),
            viewFormats.map([](auto& viewFormat) {
                return WebCore::convertToBacking(viewFormat);
            }),
            WebCore::convertToBacking(colorSpace),
            WebCore::convertToBacking(toneMapping.mode),
            WebCore::convertToBacking(alphaMode),
            reportValidationErrors,
        };
    }

    WeakPtr<GPUDevice, WeakPtrImplWithEventTargetData> device;
    GPUTextureFormat format { GPUTextureFormat::R8unorm };
    GPUTextureUsageFlags usage { GPUTextureUsage::RENDER_ATTACHMENT };
    Vector<GPUTextureFormat> viewFormats;
    GPUPredefinedColorSpace colorSpace { GPUPredefinedColorSpace::SRGB };
    GPUCanvasToneMapping toneMapping;
    GPUCanvasAlphaMode alphaMode { GPUCanvasAlphaMode::Opaque };
};

}
