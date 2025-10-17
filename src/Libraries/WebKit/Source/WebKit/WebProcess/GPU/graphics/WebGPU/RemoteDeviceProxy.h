/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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

#include "RemoteAdapterProxy.h"
#include "SharedVideoFrame.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPUCommandEncoderDescriptor.h>
#include <WebCore/WebGPUDevice.h>
#include <wtf/TZoneMalloc.h>

#if PLATFORM(COCOA) && ENABLE(VIDEO)
#include <WebCore/MediaPlayerIdentifier.h>
#endif

namespace WebKit::WebGPU {

class ConvertToBackingContext;
class RemoteQueueProxy;

class RemoteDeviceProxy final : public WebCore::WebGPU::Device {
    WTF_MAKE_TZONE_ALLOCATED(RemoteDeviceProxy);
public:
    static Ref<RemoteDeviceProxy> create(Ref<WebCore::WebGPU::SupportedFeatures>&& features, Ref<WebCore::WebGPU::SupportedLimits>&& limits, RemoteAdapterProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier, WebGPUIdentifier queueIdentifier)
    {
        return adoptRef(*new RemoteDeviceProxy(WTFMove(features), WTFMove(limits), parent, convertToBackingContext, identifier, queueIdentifier));
    }

    virtual ~RemoteDeviceProxy();

    RemoteAdapterProxy& parent() { return m_parent; }
    RemoteGPUProxy& root() { return m_parent->root(); }
    Ref<RemoteGPUProxy> protectedRoot() { return m_parent->root(); }
    WebGPUIdentifier backing() const { return m_backing; }

private:
    friend class DowncastConvertToBackingContext;

    RemoteDeviceProxy(Ref<WebCore::WebGPU::SupportedFeatures>&&, Ref<WebCore::WebGPU::SupportedLimits>&&, RemoteAdapterProxy&, ConvertToBackingContext&, WebGPUIdentifier, WebGPUIdentifier queueIdentifier);

    RemoteDeviceProxy(const RemoteDeviceProxy&) = delete;
    RemoteDeviceProxy(RemoteDeviceProxy&&) = delete;
    RemoteDeviceProxy& operator=(const RemoteDeviceProxy&) = delete;
    RemoteDeviceProxy& operator=(RemoteDeviceProxy&&) = delete;

    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(WTFMove(message), backing());
    }
    template<typename T, typename C>
    WARN_UNUSED_RETURN std::optional<IPC::StreamClientConnection::AsyncReplyID> sendWithAsyncReply(T&& message, C&& completionHandler)
    {
        return root().protectedStreamClientConnection()->sendWithAsyncReply(WTFMove(message), completionHandler, backing());
    }

    Ref<WebCore::WebGPU::Queue> queue() final;

    void destroy() final;

    RefPtr<WebCore::WebGPU::XRBinding> createXRBinding() final;
    RefPtr<WebCore::WebGPU::Buffer> createBuffer(const WebCore::WebGPU::BufferDescriptor&) final;
    RefPtr<WebCore::WebGPU::Texture> createTexture(const WebCore::WebGPU::TextureDescriptor&) final;
    RefPtr<WebCore::WebGPU::Sampler> createSampler(const WebCore::WebGPU::SamplerDescriptor&) final;
    RefPtr<WebCore::WebGPU::ExternalTexture> importExternalTexture(const WebCore::WebGPU::ExternalTextureDescriptor&) final;
#if PLATFORM(COCOA) && ENABLE(VIDEO)
    void updateExternalTexture(const WebCore::WebGPU::ExternalTexture&, const WebCore::MediaPlayerIdentifier&) final;
#endif

    RefPtr<WebCore::WebGPU::BindGroupLayout> createBindGroupLayout(const WebCore::WebGPU::BindGroupLayoutDescriptor&) final;
    RefPtr<WebCore::WebGPU::PipelineLayout> createPipelineLayout(const WebCore::WebGPU::PipelineLayoutDescriptor&) final;
    RefPtr<WebCore::WebGPU::BindGroup> createBindGroup(const WebCore::WebGPU::BindGroupDescriptor&) final;

    RefPtr<WebCore::WebGPU::ShaderModule> createShaderModule(const WebCore::WebGPU::ShaderModuleDescriptor&) final;
    RefPtr<WebCore::WebGPU::ComputePipeline> createComputePipeline(const WebCore::WebGPU::ComputePipelineDescriptor&) final;
    RefPtr<WebCore::WebGPU::RenderPipeline> createRenderPipeline(const WebCore::WebGPU::RenderPipelineDescriptor&) final;
    void createComputePipelineAsync(const WebCore::WebGPU::ComputePipelineDescriptor&, CompletionHandler<void(RefPtr<WebCore::WebGPU::ComputePipeline>&&, String&&)>&&) final;
    void createRenderPipelineAsync(const WebCore::WebGPU::RenderPipelineDescriptor&, CompletionHandler<void(RefPtr<WebCore::WebGPU::RenderPipeline>&&, String&&)>&&) final;

    RefPtr<WebCore::WebGPU::CommandEncoder> createCommandEncoder(const std::optional<WebCore::WebGPU::CommandEncoderDescriptor>&) final;
    Ref<WebCore::WebGPU::CommandEncoder> createInvalidCommandEncoder();
    RefPtr<WebCore::WebGPU::RenderBundleEncoder> createRenderBundleEncoder(const WebCore::WebGPU::RenderBundleEncoderDescriptor&) final;

    RefPtr<WebCore::WebGPU::QuerySet> createQuerySet(const WebCore::WebGPU::QuerySetDescriptor&) final;

    void pushErrorScope(WebCore::WebGPU::ErrorFilter) final;
    void popErrorScope(CompletionHandler<void(bool, std::optional<WebCore::WebGPU::Error>&&)>&&) final;
    void resolveUncapturedErrorEvent(CompletionHandler<void(bool, std::optional<WebCore::WebGPU::Error>&&)>&&) final;

    void setLabelInternal(const String&) final;
    void resolveDeviceLostPromise(CompletionHandler<void(WebCore::WebGPU::DeviceLostReason)>&&) final;

    Ref<ConvertToBackingContext> protectedConvertToBackingContext() const;

    Ref<WebCore::WebGPU::CommandEncoder> invalidCommandEncoder() final;
    Ref<WebCore::WebGPU::CommandBuffer> invalidCommandBuffer() final;
    Ref<WebCore::WebGPU::RenderPassEncoder> invalidRenderPassEncoder() final;
    Ref<WebCore::WebGPU::ComputePassEncoder> invalidComputePassEncoder() final;
    void pauseAllErrorReporting(bool pause) final;

    WebGPUIdentifier m_backing;
    Ref<ConvertToBackingContext> m_convertToBackingContext;
    Ref<RemoteAdapterProxy> m_parent;
    Ref<RemoteQueueProxy> m_queue;
#if PLATFORM(COCOA) && ENABLE(VIDEO)
    WebKit::SharedVideoFrameWriter m_sharedVideoFrameWriter;
#endif
    Ref<WebCore::WebGPU::CommandEncoder> m_invalidCommandEncoder;
    Ref<WebCore::WebGPU::RenderPassEncoder> m_invalidRenderPassEncoder;
    Ref<WebCore::WebGPU::ComputePassEncoder> m_invalidComputePassEncoder;
    Ref<WebCore::WebGPU::CommandBuffer> m_invalidCommandBuffer;
};

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
