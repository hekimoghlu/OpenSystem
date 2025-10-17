/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#include "WebGPUImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUAdapterImpl.h"
#include "WebGPUCompositorIntegrationImpl.h"
#include "WebGPUDowncastConvertToBackingContext.h"
#include "WebGPUPresentationContextDescriptor.h"
#include "WebGPUPresentationContextImpl.h"
#include <WebCore/GraphicsContext.h>
#include <WebCore/IntSize.h>
#include <WebCore/NativeImage.h>
#include <WebGPU/WebGPUExt.h>
#include <wtf/BlockPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GPUImpl);

GPUImpl::GPUImpl(WebGPUPtr<WGPUInstance>&& instance, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(instance))
    , m_convertToBackingContext(convertToBackingContext)
{
}

GPUImpl::~GPUImpl() = default;

static void requestAdapterCallback(WGPURequestAdapterStatus status, WGPUAdapter adapter, const char* message, void* userdata)
{
    auto block = reinterpret_cast<void(^)(WGPURequestAdapterStatus, WGPUAdapter, const char*)>(userdata);
    block(status, adapter, message);
    Block_release(block); // Block_release is matched with Block_copy below in GPUImpl::requestAdapter().
}

void GPUImpl::requestAdapter(const RequestAdapterOptions& options, CompletionHandler<void(RefPtr<Adapter>&&)>&& callback)
{
    Ref convertToBackingContext = m_convertToBackingContext;

    WGPURequestAdapterOptions backingOptions {
        .nextInChain = nullptr,
        .compatibleSurface = nullptr,
#if CPU(X86_64)
        .powerPreference = WGPUPowerPreference_HighPerformance,
#else
        .powerPreference = options.powerPreference ? convertToBackingContext->convertToBacking(*options.powerPreference) : static_cast<WGPUPowerPreference>(WGPUPowerPreference_Undefined),
#endif
        .backendType = WGPUBackendType_Metal,
        .forceFallbackAdapter = options.forceFallbackAdapter,
        .xrCompatible = options.xrCompatible,
    };

    auto blockPtr = makeBlockPtr([convertToBackingContext = convertToBackingContext.copyRef(), callback = WTFMove(callback)](WGPURequestAdapterStatus status, WGPUAdapter adapter, const char*) mutable {
        if (status == WGPURequestAdapterStatus_Success)
            callback(AdapterImpl::create(adoptWebGPU(adapter), convertToBackingContext));
        else
            callback(nullptr);
    });
    wgpuInstanceRequestAdapter(m_backing.get(), &backingOptions, &requestAdapterCallback, Block_copy(blockPtr.get())); // Block_copy is matched with Block_release above in requestAdapterCallback().
}

static WTF::Function<void(CompletionHandler<void()>&&)> convert(WGPUOnSubmittedWorkScheduledCallback&& onSubmittedWorkScheduledCallback)
{
    return [onSubmittedWorkScheduledCallback = makeBlockPtr(WTFMove(onSubmittedWorkScheduledCallback))](CompletionHandler<void()>&& completionHandler) {
        onSubmittedWorkScheduledCallback(makeBlockPtr(WTFMove(completionHandler)).get());
    };
}

RefPtr<PresentationContext> GPUImpl::createPresentationContext(const PresentationContextDescriptor& presentationContextDescriptor)
{
    auto& compositorIntegration = m_convertToBackingContext->convertToBacking(presentationContextDescriptor.compositorIntegration);

    auto registerCallbacksBlock = makeBlockPtr([&](WGPURenderBuffersWereRecreatedBlockCallback renderBuffersWereRecreatedCallback, WGPUOnSubmittedWorkScheduledCallback onSubmittedWorkScheduledCallback) {
        compositorIntegration.registerCallbacks(makeBlockPtr(WTFMove(renderBuffersWereRecreatedCallback)), convert(WTFMove(onSubmittedWorkScheduledCallback)));
    });

    WGPUSurfaceDescriptorCocoaCustomSurface cocoaSurface {
        {
            nullptr,
            static_cast<WGPUSType>(WGPUSTypeExtended_SurfaceDescriptorCocoaSurfaceBacking),
        },
        registerCallbacksBlock.get(),
    };

    WGPUSurfaceDescriptor surfaceDescriptor {
        &cocoaSurface.chain,
        nullptr,
    };

    auto result = PresentationContextImpl::create(adoptWebGPU(wgpuInstanceCreateSurface(m_backing.get(), &surfaceDescriptor)), m_convertToBackingContext);
    compositorIntegration.setPresentationContext(result);
    return result;
}

RefPtr<CompositorIntegration> GPUImpl::createCompositorIntegration()
{
    return CompositorIntegrationImpl::create(m_convertToBackingContext);
}

void GPUImpl::paintToCanvas(WebCore::NativeImage& image, const WebCore::IntSize& canvasSize, WebCore::GraphicsContext& context)
{
    auto imageSize = image.size();
    FloatRect canvasRect(FloatPoint(), canvasSize);
    GraphicsContextStateSaver stateSaver(context);
    context.setImageInterpolationQuality(InterpolationQuality::DoNotInterpolate);
    context.drawNativeImage(image, canvasRect, FloatRect(FloatPoint(), imageSize), { CompositeOperator::Copy });
}

bool GPUImpl::isValid(const CompositorIntegration&) const
{
    return true;
}

bool GPUImpl::isValid(const Buffer& buffer) const
{
    WGPUBuffer wgpuBuffer = m_convertToBackingContext.get().convertToBacking(buffer);
    return wgpuBufferIsValid(wgpuBuffer);
}

bool GPUImpl::isValid(const Adapter& adapter) const
{
    WGPUAdapter wgpuAdapter = m_convertToBackingContext.get().convertToBacking(adapter);
    return wgpuAdapterIsValid(wgpuAdapter);
}

bool GPUImpl::isValid(const BindGroup& bindGroup) const
{
    WGPUBindGroup wgpuBindGroup = m_convertToBackingContext.get().convertToBacking(bindGroup);
    return wgpuBindGroupIsValid(wgpuBindGroup);
}

bool GPUImpl::isValid(const BindGroupLayout& bindGroupLayout) const
{
    WGPUBindGroupLayout wgpuBindGroupLayout = m_convertToBackingContext.get().convertToBacking(bindGroupLayout);
    return wgpuBindGroupLayoutIsValid(wgpuBindGroupLayout);
}

bool GPUImpl::isValid(const CommandBuffer& commandBuffer) const
{
    WGPUCommandBuffer wgpuCommandBuffer = m_convertToBackingContext.get().convertToBacking(commandBuffer);
    return wgpuCommandBufferIsValid(wgpuCommandBuffer);
}

bool GPUImpl::isValid(const CommandEncoder& commandEncoder) const
{
    WGPUCommandEncoder wgpuCommandEncoder = m_convertToBackingContext.get().convertToBacking(commandEncoder);
    return wgpuCommandEncoderIsValid(wgpuCommandEncoder);
}

bool GPUImpl::isValid(const ComputePassEncoder& computePassEncoder) const
{
    WGPUComputePassEncoder wgpuComputePassEncoder = m_convertToBackingContext.get().convertToBacking(computePassEncoder);
    return wgpuComputePassEncoderIsValid(wgpuComputePassEncoder);
}

bool GPUImpl::isValid(const ComputePipeline& computePipeline) const
{
    WGPUComputePipeline wgpuComputePipeline = m_convertToBackingContext.get().convertToBacking(computePipeline);
    return wgpuComputePipelineIsValid(wgpuComputePipeline);
}

bool GPUImpl::isValid(const Device& device) const
{
    WGPUDevice wgpuDevice = m_convertToBackingContext.get().convertToBacking(device);
    return wgpuDeviceIsValid(wgpuDevice);
}

bool GPUImpl::isValid(const ExternalTexture& externalTexture) const
{
    WGPUExternalTexture wgpuExternalTexture = m_convertToBackingContext.get().convertToBacking(externalTexture);
    return wgpuExternalTextureIsValid(wgpuExternalTexture);
}

bool GPUImpl::isValid(const PipelineLayout& pipelineLayout) const
{
    WGPUPipelineLayout wgpuPipelineLayout = m_convertToBackingContext.get().convertToBacking(pipelineLayout);
    return wgpuPipelineLayoutIsValid(wgpuPipelineLayout);
}

bool GPUImpl::isValid(const PresentationContext& presentationContext) const
{
    WGPUSurface wgpuPresentationContext = m_convertToBackingContext.get().convertToBacking(presentationContext);
    return wgpuPresentationContextIsValid(wgpuPresentationContext);
}

bool GPUImpl::isValid(const QuerySet& querySet) const
{
    WGPUQuerySet wgpuQuerySet = m_convertToBackingContext.get().convertToBacking(querySet);
    return wgpuQuerySetIsValid(wgpuQuerySet);
}

bool GPUImpl::isValid(const Queue& queue) const
{
    WGPUQueue wgpuQueue = m_convertToBackingContext.get().convertToBacking(queue);
    return wgpuQueueIsValid(wgpuQueue);
}

bool GPUImpl::isValid(const RenderBundleEncoder& renderBundleEncoder) const
{
    WGPURenderBundleEncoder wgpuRenderBundleEncoder = m_convertToBackingContext.get().convertToBacking(renderBundleEncoder);
    return wgpuRenderBundleEncoderIsValid(wgpuRenderBundleEncoder);
}

bool GPUImpl::isValid(const RenderBundle& renderBundle) const
{
    WGPURenderBundle wgpuRenderBundle = m_convertToBackingContext.get().convertToBacking(renderBundle);
    return wgpuRenderBundleIsValid(wgpuRenderBundle);
}

bool GPUImpl::isValid(const RenderPassEncoder& renderPassEncoder) const
{
    WGPURenderPassEncoder wgpuRenderPassEncoder = m_convertToBackingContext.get().convertToBacking(renderPassEncoder);
    return wgpuRenderPassEncoderIsValid(wgpuRenderPassEncoder);
}

bool GPUImpl::isValid(const RenderPipeline& renderPipeline) const
{
    WGPURenderPipeline wgpuRenderPipeline = m_convertToBackingContext.get().convertToBacking(renderPipeline);
    return wgpuRenderPipelineIsValid(wgpuRenderPipeline);
}

bool GPUImpl::isValid(const Sampler& sampler) const
{
    WGPUSampler wgpuSampler = m_convertToBackingContext.get().convertToBacking(sampler);
    return wgpuSamplerIsValid(wgpuSampler);
}

bool GPUImpl::isValid(const ShaderModule& shaderModule) const
{
    WGPUShaderModule wgpuShaderModule = m_convertToBackingContext.get().convertToBacking(shaderModule);
    return wgpuShaderModuleIsValid(wgpuShaderModule);
}

bool GPUImpl::isValid(const Texture& texture) const
{
    WGPUTexture wgpuTexture = m_convertToBackingContext.get().convertToBacking(texture);
    return wgpuTextureIsValid(wgpuTexture);
}

bool GPUImpl::isValid(const TextureView& textureView) const
{
    WGPUTextureView wgpuTextureView = m_convertToBackingContext.get().convertToBacking(textureView);
    return wgpuTextureViewIsValid(wgpuTextureView);
}

bool GPUImpl::isValid(const XRBinding& binding) const
{
    WGPUXRBinding wgpuBinding = m_convertToBackingContext.get().convertToBacking(binding);
    return wgpuXRBindingIsValid(wgpuBinding);
}

bool GPUImpl::isValid(const XRSubImage& subImage) const
{
    WGPUXRSubImage wgpuSubImage = m_convertToBackingContext.get().convertToBacking(subImage);
    return wgpuXRSubImageIsValid(wgpuSubImage);
}

bool GPUImpl::isValid(const XRProjectionLayer& layer) const
{
    WGPUXRProjectionLayer wgpuLayer = m_convertToBackingContext.get().convertToBacking(layer);
    return wgpuXRProjectionLayerIsValid(wgpuLayer);
}

bool GPUImpl::isValid(const XRView& view) const
{
    WGPUXRView wgpuView = m_convertToBackingContext.get().convertToBacking(view);
    return wgpuXRViewIsValid(wgpuView);
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
