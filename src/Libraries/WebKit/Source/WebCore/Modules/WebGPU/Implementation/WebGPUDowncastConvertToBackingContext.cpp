/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
#include "config.h"
#include "WebGPUDowncastConvertToBackingContext.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUAdapterImpl.h"
#include "WebGPUBindGroupImpl.h"
#include "WebGPUBindGroupLayoutImpl.h"
#include "WebGPUBufferImpl.h"
#include "WebGPUCommandBufferImpl.h"
#include "WebGPUCommandEncoderImpl.h"
#include "WebGPUCompositorIntegrationImpl.h"
#include "WebGPUComputePassEncoderImpl.h"
#include "WebGPUComputePipelineImpl.h"
#include "WebGPUDeviceImpl.h"
#include "WebGPUExternalTextureImpl.h"
#include "WebGPUImpl.h"
#include "WebGPUPipelineLayoutImpl.h"
#include "WebGPUPresentationContextImpl.h"
#include "WebGPUQuerySetImpl.h"
#include "WebGPUQueueImpl.h"
#include "WebGPURenderBundleEncoderImpl.h"
#include "WebGPURenderBundleImpl.h"
#include "WebGPURenderPassEncoderImpl.h"
#include "WebGPURenderPipelineImpl.h"
#include "WebGPUSamplerImpl.h"
#include "WebGPUShaderModuleImpl.h"
#include "WebGPUTextureImpl.h"
#include "WebGPUTextureViewImpl.h"
#include "WebGPUXRBindingImpl.h"
#include "WebGPUXRProjectionLayerImpl.h"
#include "WebGPUXRSubImageImpl.h"
#include "WebGPUXRViewImpl.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DowncastConvertToBackingContext);

WGPUAdapter DowncastConvertToBackingContext::convertToBacking(const Adapter& adapter)
{
    return static_cast<const AdapterImpl&>(adapter).backing();
}

WGPUBindGroup DowncastConvertToBackingContext::convertToBacking(const BindGroup& bindGroup)
{
    return static_cast<const BindGroupImpl&>(bindGroup).backing();
}

WGPUBindGroupLayout DowncastConvertToBackingContext::convertToBacking(const BindGroupLayout& bindGroupLayout)
{
    return static_cast<const BindGroupLayoutImpl&>(bindGroupLayout).backing();
}

WGPUBuffer DowncastConvertToBackingContext::convertToBacking(const Buffer& buffer)
{
    return static_cast<const BufferImpl&>(buffer).backing();
}

WGPUCommandBuffer DowncastConvertToBackingContext::convertToBacking(const CommandBuffer& commandBuffer)
{
    return static_cast<const CommandBufferImpl&>(commandBuffer).backing();
}

WGPUCommandEncoder DowncastConvertToBackingContext::convertToBacking(const CommandEncoder& commandEncoder)
{
    return static_cast<const CommandEncoderImpl&>(commandEncoder).backing();
}

WGPUComputePassEncoder DowncastConvertToBackingContext::convertToBacking(const ComputePassEncoder& computePassEncoder)
{
    return static_cast<const ComputePassEncoderImpl&>(computePassEncoder).backing();
}

WGPUComputePipeline DowncastConvertToBackingContext::convertToBacking(const ComputePipeline& computePipeline)
{
    return static_cast<const ComputePipelineImpl&>(computePipeline).backing();
}

WGPUDevice DowncastConvertToBackingContext::convertToBacking(const Device& device)
{
    return static_cast<const DeviceImpl&>(device).backing();
}

WGPUExternalTexture DowncastConvertToBackingContext::convertToBacking(const ExternalTexture& externalTexture)
{
    return static_cast<const ExternalTextureImpl&>(externalTexture).backing();
}

WGPUInstance DowncastConvertToBackingContext::convertToBacking(const GPU& gpu)
{
    return static_cast<const GPUImpl&>(gpu).backing();
}

WGPUPipelineLayout DowncastConvertToBackingContext::convertToBacking(const PipelineLayout& pipelineLayout)
{
    return static_cast<const PipelineLayoutImpl&>(pipelineLayout).backing();
}

WGPUSurface DowncastConvertToBackingContext::convertToBacking(const PresentationContext& presentationContext)
{
    return static_cast<const PresentationContextImpl&>(presentationContext).backing();
}

WGPUQuerySet DowncastConvertToBackingContext::convertToBacking(const QuerySet& querySet)
{
    return static_cast<const QuerySetImpl&>(querySet).backing();
}

WGPUQueue DowncastConvertToBackingContext::convertToBacking(const Queue& queue)
{
    return static_cast<const QueueImpl&>(queue).backing();
}

WGPURenderBundleEncoder DowncastConvertToBackingContext::convertToBacking(const RenderBundleEncoder& renderBundleEncoder)
{
    return static_cast<const RenderBundleEncoderImpl&>(renderBundleEncoder).backing();
}

WGPURenderBundle DowncastConvertToBackingContext::convertToBacking(const RenderBundle& renderBundle)
{
    return static_cast<const RenderBundleImpl&>(renderBundle).backing();
}

WGPURenderPassEncoder DowncastConvertToBackingContext::convertToBacking(const RenderPassEncoder& renderPassEncoder)
{
    return static_cast<const RenderPassEncoderImpl&>(renderPassEncoder).backing();
}

WGPURenderPipeline DowncastConvertToBackingContext::convertToBacking(const RenderPipeline& renderPipeline)
{
    return static_cast<const RenderPipelineImpl&>(renderPipeline).backing();
}

WGPUSampler DowncastConvertToBackingContext::convertToBacking(const Sampler& sampler)
{
    return static_cast<const SamplerImpl&>(sampler).backing();
}

WGPUShaderModule DowncastConvertToBackingContext::convertToBacking(const ShaderModule& shaderModule)
{
    return static_cast<const ShaderModuleImpl&>(shaderModule).backing();
}

WGPUTexture DowncastConvertToBackingContext::convertToBacking(const Texture& texture)
{
    return static_cast<const TextureImpl&>(texture).backing();
}

WGPUTextureView DowncastConvertToBackingContext::convertToBacking(const TextureView& textureView)
{
    return static_cast<const TextureViewImpl&>(textureView).backing();
}

CompositorIntegrationImpl& DowncastConvertToBackingContext::convertToBacking(CompositorIntegration& compositorIntegration)
{
    return static_cast<CompositorIntegrationImpl&>(compositorIntegration);
}

WGPUXRBinding DowncastConvertToBackingContext::convertToBacking(const XRBinding& xrBinding)
{
    return static_cast<const XRBindingImpl&>(xrBinding).backing();
}

WGPUXRProjectionLayer DowncastConvertToBackingContext::convertToBacking(const XRProjectionLayer& layer)
{
    return static_cast<const XRProjectionLayerImpl&>(layer).backing();
}

WGPUXRSubImage DowncastConvertToBackingContext::convertToBacking(const XRSubImage& subImage)
{
    return static_cast<const XRSubImageImpl&>(subImage).backing();
}

WGPUXRView DowncastConvertToBackingContext::convertToBacking(const XRView& xrView)
{
    return static_cast<const XRViewImpl&>(xrView).backing();
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
