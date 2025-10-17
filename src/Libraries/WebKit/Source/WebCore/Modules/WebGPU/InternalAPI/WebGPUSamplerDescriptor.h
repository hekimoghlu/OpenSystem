/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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

#include "WebGPUAddressMode.h"
#include "WebGPUCompareFunction.h"
#include "WebGPUFilterMode.h"
#include "WebGPUObjectDescriptorBase.h"
#include <cstdint>
#include <optional>

namespace WebCore::WebGPU {

struct SamplerDescriptor : public ObjectDescriptorBase {
    AddressMode addressModeU { AddressMode::ClampToEdge };
    AddressMode addressModeV { AddressMode::ClampToEdge };
    AddressMode addressModeW { AddressMode::ClampToEdge };
    FilterMode magFilter { FilterMode::Nearest };
    FilterMode minFilter { FilterMode::Nearest };
    MipmapFilterMode mipmapFilter { MipmapFilterMode::Nearest };
    float lodMinClamp { 0 };
    float lodMaxClamp { 32 };
    std::optional<CompareFunction> compare;
    uint16_t maxAnisotropy { 1 };
};

} // namespace WebCore::WebGPU
