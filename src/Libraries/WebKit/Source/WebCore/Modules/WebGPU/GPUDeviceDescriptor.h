/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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

#include "GPUFeatureName.h"
#include "GPUObjectDescriptorBase.h"
#include "GPUQueueDescriptor.h"
#include "WebGPUDeviceDescriptor.h"
#include <cstdint>
#include <wtf/HashMap.h>
#include <wtf/KeyValuePair.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

struct GPUDeviceDescriptor : public GPUObjectDescriptorBase {
    WebGPU::DeviceDescriptor convertToBacking() const
    {
        return {
            { label },
            requiredFeatures.map([](const auto& requiredFeature) {
                return WebCore::convertToBacking(requiredFeature);
            }),
            requiredLimits,
        };
    }

    Vector<GPUFeatureName> requiredFeatures;
    Vector<KeyValuePair<String, uint64_t>> requiredLimits;
    GPUQueueDescriptor defaultQueue;
};

}
