/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 14, 2023.
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

#if ENABLE(GPU_PROCESS)

#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUCanvasAlphaMode.h>
#include <WebCore/WebGPUCanvasToneMappingMode.h>
#include <WebCore/WebGPUPredefinedColorSpace.h>
#include <WebCore/WebGPUTextureFormat.h>
#include <WebCore/WebGPUTextureUsage.h>
#include <optional>
#include <wtf/Ref.h>

namespace WebKit::WebGPU {

class Device;

struct CanvasConfiguration {
    WebGPUIdentifier device;
    WebCore::WebGPU::TextureFormat format { WebCore::WebGPU::TextureFormat::R8unorm };
    WebCore::WebGPU::TextureUsageFlags usage { WebCore::WebGPU::TextureUsage::RenderAttachment };
    Vector<WebCore::WebGPU::TextureFormat> viewFormats;
    WebCore::WebGPU::PredefinedColorSpace colorSpace { WebCore::WebGPU::PredefinedColorSpace::SRGB };
    WebCore::WebGPU::CanvasToneMappingMode toneMappingMode { WebCore::WebGPU::CanvasToneMappingMode::Standard };
    WebCore::WebGPU::CanvasAlphaMode compositingAlphaMode { WebCore::WebGPU::CanvasAlphaMode::Opaque };
    bool reportValidationErrors { true };
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
