/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 23, 2025.
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

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class DowncastConvertToBackingContext final : public ConvertToBackingContext {
    WTF_MAKE_TZONE_ALLOCATED(DowncastConvertToBackingContext);
public:
    static Ref<DowncastConvertToBackingContext> create()
    {
        return adoptRef(*new DowncastConvertToBackingContext());
    }

    virtual ~DowncastConvertToBackingContext() = default;

    WGPUAdapter convertToBacking(const Adapter&) final;
    WGPUBindGroup convertToBacking(const BindGroup&) final;
    WGPUBindGroupLayout convertToBacking(const BindGroupLayout&) final;
    WGPUBuffer convertToBacking(const Buffer&) final;
    WGPUCommandBuffer convertToBacking(const CommandBuffer&) final;
    WGPUCommandEncoder convertToBacking(const CommandEncoder&) final;
    WGPUComputePassEncoder convertToBacking(const ComputePassEncoder&) final;
    WGPUComputePipeline convertToBacking(const ComputePipeline&) final;
    WGPUDevice convertToBacking(const Device&) final;
    WGPUExternalTexture convertToBacking(const ExternalTexture&) final;
    WGPUInstance convertToBacking(const GPU&) final;
    WGPUPipelineLayout convertToBacking(const PipelineLayout&) final;
    WGPUSurface convertToBacking(const PresentationContext&) final;
    WGPUQuerySet convertToBacking(const QuerySet&) final;
    WGPUQueue convertToBacking(const Queue&) final;
    WGPURenderBundleEncoder convertToBacking(const RenderBundleEncoder&) final;
    WGPURenderBundle convertToBacking(const RenderBundle&) final;
    WGPURenderPassEncoder convertToBacking(const RenderPassEncoder&) final;
    WGPURenderPipeline convertToBacking(const RenderPipeline&) final;
    WGPUSampler convertToBacking(const Sampler&) final;
    WGPUShaderModule convertToBacking(const ShaderModule&) final;
    WGPUTexture convertToBacking(const Texture&) final;
    WGPUTextureView convertToBacking(const TextureView&) final;
    CompositorIntegrationImpl& convertToBacking(CompositorIntegration&) final;
    WGPUXRBinding convertToBacking(const XRBinding&) final;
    WGPUXRProjectionLayer convertToBacking(const XRProjectionLayer&) final;
    WGPUXRSubImage convertToBacking(const XRSubImage&) final;
    WGPUXRView convertToBacking(const XRView&) final;

private:
    DowncastConvertToBackingContext() = default;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
