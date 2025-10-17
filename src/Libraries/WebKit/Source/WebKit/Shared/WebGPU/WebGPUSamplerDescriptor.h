/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 8, 2024.
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

#include "WebGPUObjectDescriptorBase.h"
#include <WebCore/WebGPUAddressMode.h>
#include <WebCore/WebGPUCompareFunction.h>
#include <WebCore/WebGPUFilterMode.h>
#include <cstdint>
#include <optional>

namespace WebKit::WebGPU {

struct SamplerDescriptor : public ObjectDescriptorBase {
    WebCore::WebGPU::AddressMode addressModeU { WebCore::WebGPU::AddressMode::ClampToEdge };
    WebCore::WebGPU::AddressMode addressModeV { WebCore::WebGPU::AddressMode::ClampToEdge };
    WebCore::WebGPU::AddressMode addressModeW { WebCore::WebGPU::AddressMode::ClampToEdge };
    WebCore::WebGPU::FilterMode magFilter { WebCore::WebGPU::FilterMode::Nearest };
    WebCore::WebGPU::FilterMode minFilter { WebCore::WebGPU::FilterMode::Nearest };
    WebCore::WebGPU::MipmapFilterMode mipmapFilter { WebCore::WebGPU::MipmapFilterMode::Nearest };
    float lodMinClamp { 0 };
    float lodMaxClamp { 32 };
    std::optional<WebCore::WebGPU::CompareFunction> compare;
    uint16_t maxAnisotropy { 1 };
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
