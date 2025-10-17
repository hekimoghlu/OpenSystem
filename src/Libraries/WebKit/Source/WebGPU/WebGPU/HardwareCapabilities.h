/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#include "WebGPU.h"
#include "WebGPUExt.h"
#include <Metal/Metal.h>
#include <optional>
#include <wtf/Vector.h>

namespace WebGPU {

struct HardwareCapabilities {
    WGPULimits limits { };
    Vector<WGPUFeatureName> features;

    struct BaseCapabilities {
        MTLArgumentBuffersTier argumentBuffersTier { MTLArgumentBuffersTier1 };
        bool supportsNonPrivateDepthStencilTextures { false };
        id<MTLCounterSet> timestampCounterSet { nil };
        id<MTLCounterSet> statisticCounterSet { nil };
        // FIXME: canPresentRGB10A2PixelFormats isn't actually a _hardware_ capability,
        // as all hardware can render to this format. It's unclear whether this should
        // apply to _all_ PresentationContexts or just PresentationContextCoreAnimation.
        bool canPresentRGB10A2PixelFormats { false };
    } baseCapabilities;
};

std::optional<HardwareCapabilities> hardwareCapabilities(id<MTLDevice>);
bool isValid(const WGPULimits&);
WGPULimits defaultLimits();
bool anyLimitIsBetterThan(const WGPULimits& target, const WGPULimits& reference);
bool includesUnsupportedFeatures(const Vector<WGPUFeatureName>& target, const Vector<WGPUFeatureName>& reference);

} // namespace WebGPU
