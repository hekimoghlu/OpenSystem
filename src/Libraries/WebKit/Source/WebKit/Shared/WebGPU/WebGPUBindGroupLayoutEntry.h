/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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

#include "WebGPUBufferBindingLayout.h"
#include "WebGPUExternalTextureBindingLayout.h"
#include "WebGPUSamplerBindingLayout.h"
#include "WebGPUStorageTextureBindingLayout.h"
#include "WebGPUTextureBindingLayout.h"
#include <WebCore/WebGPUIntegralTypes.h>
#include <WebCore/WebGPUShaderStage.h>
#include <optional>

namespace WebKit::WebGPU {

struct BindGroupLayoutEntry {
    WebCore::WebGPU::Index32 binding { 0 };
    WebCore::WebGPU::ShaderStageFlags visibility;

    std::optional<BufferBindingLayout> buffer;
    std::optional<SamplerBindingLayout> sampler;
    std::optional<TextureBindingLayout> texture;
    std::optional<StorageTextureBindingLayout> storageTexture;
    std::optional<ExternalTextureBindingLayout> externalTexture;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
