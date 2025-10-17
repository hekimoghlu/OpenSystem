/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 15, 2025.
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

#include "WebGPUExtent3D.h"
#include "WebGPUIntegralTypes.h"
#include "WebGPUObjectDescriptorBase.h"
#include "WebGPUTextureDimension.h"
#include "WebGPUTextureFormat.h"
#include "WebGPUTextureUsage.h"

namespace WebCore::WebGPU {

struct TextureDescriptor : public ObjectDescriptorBase {
    Extent3D size;
    IntegerCoordinate mipLevelCount { 1 };
    Size32 sampleCount { 1 };
    TextureDimension dimension { TextureDimension::_2d };
    TextureFormat format { TextureFormat::R8unorm };
    TextureUsageFlags usage;
    Vector<TextureFormat> viewFormats;
};

} // namespace WebCore::WebGPU
