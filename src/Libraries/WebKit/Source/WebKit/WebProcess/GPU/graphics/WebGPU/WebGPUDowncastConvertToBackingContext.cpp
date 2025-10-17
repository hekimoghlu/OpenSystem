/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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

#if ENABLE(GPU_PROCESS)

#include "RemoteAdapterProxy.h"
#include "RemoteBindGroupLayoutProxy.h"
#include "RemoteBindGroupProxy.h"
#include "RemoteBufferProxy.h"
#include "RemoteCommandBufferProxy.h"
#include "RemoteCommandEncoderProxy.h"
#include "RemoteCompositorIntegrationProxy.h"
#include "RemoteComputePassEncoderProxy.h"
#include "RemoteComputePipelineProxy.h"
#include "RemoteDeviceProxy.h"
#include "RemoteExternalTextureProxy.h"
#include "RemoteGPUProxy.h"
#include "RemotePipelineLayoutProxy.h"
#include "RemotePresentationContextProxy.h"
#include "RemoteQuerySetProxy.h"
#include "RemoteQueueProxy.h"
#include "RemoteRenderBundleEncoderProxy.h"
#include "RemoteRenderBundleProxy.h"
#include "RemoteRenderPassEncoderProxy.h"
#include "RemoteRenderPipelineProxy.h"
#include "RemoteSamplerProxy.h"
#include "RemoteShaderModuleProxy.h"
#include "RemoteTextureProxy.h"
#include "RemoteTextureViewProxy.h"
#include "RemoteXRBindingProxy.h"
#include "RemoteXRProjectionLayerProxy.h"
#include "RemoteXRSubImageProxy.h"
#include "RemoteXRViewProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DowncastConvertToBackingContext);

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Adapter& adapter)
{
    return static_cast<const RemoteAdapterProxy&>(adapter).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::BindGroup& bindGroup)
{
    return static_cast<const RemoteBindGroupProxy&>(bindGroup).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::BindGroupLayout& bindGroupLayout)
{
    return static_cast<const RemoteBindGroupLayoutProxy&>(bindGroupLayout).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Buffer& buffer)
{
    return static_cast<const RemoteBufferProxy&>(buffer).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::CommandBuffer& commandBuffer)
{
    return static_cast<const RemoteCommandBufferProxy&>(commandBuffer).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::CommandEncoder& commandEncoder)
{
    return static_cast<const RemoteCommandEncoderProxy&>(commandEncoder).backing();
}

const RemoteCompositorIntegrationProxy& DowncastConvertToBackingContext::convertToRawBacking(const WebCore::WebGPU::CompositorIntegration& compositorIntegration)
{
    return static_cast<const RemoteCompositorIntegrationProxy&>(compositorIntegration);
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::CompositorIntegration& compositorIntegration)
{
    return static_cast<const RemoteCompositorIntegrationProxy&>(compositorIntegration).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ComputePassEncoder& computePassEncoder)
{
    return static_cast<const RemoteComputePassEncoderProxy&>(computePassEncoder).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ComputePipeline& computePipeline)
{
    return static_cast<const RemoteComputePipelineProxy&>(computePipeline).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Device& device)
{
    return static_cast<const RemoteDeviceProxy&>(device).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ExternalTexture& externalTexture)
{
    return static_cast<const RemoteExternalTextureProxy&>(externalTexture).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::GPU& gpu)
{
    return static_cast<const RemoteGPUProxy&>(gpu).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::PipelineLayout& pipelineLayout)
{
    return static_cast<const RemotePipelineLayoutProxy&>(pipelineLayout).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::PresentationContext& presentationContext)
{
    return static_cast<const RemotePresentationContextProxy&>(presentationContext).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::QuerySet& querySet)
{
    return static_cast<const RemoteQuerySetProxy&>(querySet).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Queue& queue)
{
    return static_cast<const RemoteQueueProxy&>(queue).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderBundleEncoder& renderBundleEncoder)
{
    return static_cast<const RemoteRenderBundleEncoderProxy&>(renderBundleEncoder).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderBundle& renderBundle)
{
    return static_cast<const RemoteRenderBundleProxy&>(renderBundle).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderPassEncoder& renderPassEncoder)
{
    return static_cast<const RemoteRenderPassEncoderProxy&>(renderPassEncoder).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::RenderPipeline& renderPipeline)
{
    return static_cast<const RemoteRenderPipelineProxy&>(renderPipeline).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Sampler& sampler)
{
    return static_cast<const RemoteSamplerProxy&>(sampler).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::ShaderModule& shaderModule)
{
    return static_cast<const RemoteShaderModuleProxy&>(shaderModule).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::Texture& texture)
{
    return static_cast<const RemoteTextureProxy&>(texture).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::TextureView& textureView)
{
    return static_cast<const RemoteTextureViewProxy&>(textureView).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::XRBinding& xrBinding)
{
    return static_cast<const RemoteXRBindingProxy&>(xrBinding).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::XRProjectionLayer& layer)
{
    return static_cast<const RemoteXRProjectionLayerProxy&>(layer).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::XRSubImage& subImage)
{
    return static_cast<const RemoteXRSubImageProxy&>(subImage).backing();
}

WebGPUIdentifier DowncastConvertToBackingContext::convertToBacking(const WebCore::WebGPU::XRView& view)
{
    return static_cast<const RemoteXRViewProxy&>(view).backing();
}

} // namespace WebKit::WebGPU

#endif // HAVE(GPU_PROCESS)
