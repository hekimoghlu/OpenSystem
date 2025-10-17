/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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

#include "GPUExtent3DDict.h"
#include "GPUIntegralTypes.h"
#include "GPUObjectDescriptorBase.h"
#include "GPUTextureDimension.h"
#include "GPUTextureFormat.h"
#include "GPUTextureUsage.h"
#include "WebGPUTextureDescriptor.h"

namespace WebCore {

struct GPUTextureDescriptor : public GPUObjectDescriptorBase {
    WebGPU::TextureDescriptor convertToBacking() const
    {
        return {
            { label },
            WebCore::convertToBacking(size),
            mipLevelCount,
            sampleCount,
            WebCore::convertToBacking(dimension),
            WebCore::convertToBacking(format),
            convertTextureUsageFlagsToBacking(usage),
            viewFormats.map([](auto viewFormat) {
                return WebCore::convertToBacking(viewFormat);
            }),
        };
    }

    GPUExtent3D size;
    GPUIntegerCoordinate mipLevelCount { 1 };
    GPUSize32 sampleCount { 1 };
    GPUTextureDimension dimension { GPUTextureDimension::_2d };
    GPUTextureFormat format { GPUTextureFormat::R8unorm };
    GPUTextureUsageFlags usage { 0 };
    Vector<GPUTextureFormat> viewFormats;
};

}
