/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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

#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMalloc.h>

namespace WebKit::WebGPU {

class DowncastConvertToBackingContext final : public ConvertToBackingContext {
    WTF_MAKE_TZONE_ALLOCATED(DowncastConvertToBackingContext);
public:
    static Ref<DowncastConvertToBackingContext> create()
    {
        return adoptRef(*new DowncastConvertToBackingContext());
    }

    virtual ~DowncastConvertToBackingContext() = default;

    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Adapter&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::BindGroup&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::BindGroupLayout&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Buffer&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::CommandBuffer&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::CommandEncoder&) final;
    const RemoteCompositorIntegrationProxy& convertToRawBacking(const WebCore::WebGPU::CompositorIntegration&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::CompositorIntegration&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ComputePassEncoder&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ComputePipeline&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Device&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ExternalTexture&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::GPU&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::PipelineLayout&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::PresentationContext&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::QuerySet&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Queue&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderBundleEncoder&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderBundle&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderPassEncoder&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::RenderPipeline&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Sampler&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::ShaderModule&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::Texture&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::TextureView&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRBinding&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRProjectionLayer&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRSubImage&) final;
    WebGPUIdentifier convertToBacking(const WebCore::WebGPU::XRView&) final;

private:
    DowncastConvertToBackingContext() = default;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
