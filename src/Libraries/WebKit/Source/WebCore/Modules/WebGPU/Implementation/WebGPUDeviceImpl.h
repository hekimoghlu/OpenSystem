/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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

#include "WebGPUDevice.h"
#include "WebGPUPtr.h"
#include "WebGPUQueueImpl.h"
#include <WebCore/MediaPlayerIdentifier.h>
#include <WebGPU/WebGPU.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore::WebGPU {

class ConvertToBackingContext;
enum class DeviceLostReason : uint8_t;

class DeviceImpl final : public Device {
    WTF_MAKE_TZONE_ALLOCATED(DeviceImpl);
public:
    static Ref<DeviceImpl> create(WebGPUPtr<WGPUDevice>&& device, Ref<SupportedFeatures>&& features, Ref<SupportedLimits>&& limits, ConvertToBackingContext& convertToBackingContext)
    {
        return adoptRef(*new DeviceImpl(WTFMove(device), WTFMove(features), WTFMove(limits), convertToBackingContext));
    }

    virtual ~DeviceImpl();
    void setLastUncapturedError(WGPUErrorType, char const*);

private:
    friend class DowncastConvertToBackingContext;

    DeviceImpl(WebGPUPtr<WGPUDevice>&&, Ref<SupportedFeatures>&&, Ref<SupportedLimits>&&, ConvertToBackingContext&);

    DeviceImpl(const DeviceImpl&) = delete;
    DeviceImpl(DeviceImpl&&) = delete;
    DeviceImpl& operator=(const DeviceImpl&) = delete;
    DeviceImpl& operator=(DeviceImpl&&) = delete;

    WGPUDevice backing() const { return m_backing.get(); }

    Ref<Queue> queue() final;

    void destroy() final;

    RefPtr<XRBinding> createXRBinding() final;
    RefPtr<Buffer> createBuffer(const BufferDescriptor&) final;
    RefPtr<Texture> createTexture(const TextureDescriptor&) final;
    RefPtr<Sampler> createSampler(const SamplerDescriptor&) final;
    RefPtr<ExternalTexture> importExternalTexture(const ExternalTextureDescriptor&) final;
    void updateExternalTexture(const WebCore::WebGPU::ExternalTexture&, const WebCore::MediaPlayerIdentifier&) final;

    RefPtr<BindGroupLayout> createBindGroupLayout(const BindGroupLayoutDescriptor&) final;
    RefPtr<PipelineLayout> createPipelineLayout(const PipelineLayoutDescriptor&) final;
    RefPtr<BindGroup> createBindGroup(const BindGroupDescriptor&) final;

    RefPtr<ShaderModule> createShaderModule(const ShaderModuleDescriptor&) final;
    RefPtr<ComputePipeline> createComputePipeline(const ComputePipelineDescriptor&) final;
    RefPtr<RenderPipeline> createRenderPipeline(const RenderPipelineDescriptor&) final;
    void createComputePipelineAsync(const ComputePipelineDescriptor&, CompletionHandler<void(RefPtr<ComputePipeline>&&, String&&)>&&) final;
    void createRenderPipelineAsync(const RenderPipelineDescriptor&, CompletionHandler<void(RefPtr<RenderPipeline>&&, String&&)>&&) final;

    RefPtr<CommandEncoder> createCommandEncoder(const std::optional<CommandEncoderDescriptor>&) final;
    RefPtr<RenderBundleEncoder> createRenderBundleEncoder(const RenderBundleEncoderDescriptor&) final;

    RefPtr<QuerySet> createQuerySet(const QuerySetDescriptor&) final;

    void pushErrorScope(ErrorFilter) final;
    void popErrorScope(CompletionHandler<void(bool, std::optional<Error>&&)>&&) final;
    void resolveUncapturedErrorEvent(CompletionHandler<void(bool, std::optional<WebCore::WebGPU::Error>&&)>&&) final;
    void resolveDeviceLostPromise(CompletionHandler<void(WebCore::WebGPU::DeviceLostReason)>&&) final;

    void setLabelInternal(const String&) final;
    void pauseAllErrorReporting(bool pause) final;

    [[noreturn]] Ref<CommandEncoder> invalidCommandEncoder() final;
    [[noreturn]] Ref<CommandBuffer> invalidCommandBuffer() final;
    [[noreturn]] Ref<RenderPassEncoder> invalidRenderPassEncoder() final;
    [[noreturn]] Ref<ComputePassEncoder> invalidComputePassEncoder() final;

    WebGPUPtr<WGPUDevice> m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<QueueImpl> m_queue;
};

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
