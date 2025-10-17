/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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

#include "GPUProcessConnection.h"
#include "RenderingBackendIdentifier.h"
#include "StreamClientConnection.h"
#include "WebGPUIdentifier.h"
#include <WebCore/WebGPU.h>
#include <WebCore/WebGPUPresentationContext.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {
class IntSize;
class GraphicsContext;
class NativeImage;
}

namespace WebKit {
class RemoteRenderingBackendProxy;
class WebPage;

namespace WebGPU {
class ConvertToBackingContext;
class DowncastConvertToBackingContext;
}

class RemoteGPUProxy final : public WebCore::WebGPU::GPU, private IPC::Connection::Client, public ThreadSafeRefCounted<RemoteGPUProxy>, SerialFunctionDispatcher {
    WTF_MAKE_TZONE_ALLOCATED(RemoteGPUProxy);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteGPUProxy);
public:
    static RefPtr<RemoteGPUProxy> create(WebGPU::ConvertToBackingContext&, WebPage&);
    static RefPtr<RemoteGPUProxy> create(WebGPU::ConvertToBackingContext&, RemoteRenderingBackendProxy&, SerialFunctionDispatcher&);

    virtual ~RemoteGPUProxy();

    RemoteGPUProxy& root() { return *this; }

    IPC::StreamClientConnection& streamClientConnection() { return *m_streamConnection; }
    Ref<IPC::StreamClientConnection> protectedStreamClientConnection() { return *m_streamConnection; }

    void ref() const final { return ThreadSafeRefCounted<RemoteGPUProxy>::ref(); }
    void deref() const final { return ThreadSafeRefCounted<RemoteGPUProxy>::deref(); }

    void paintToCanvas(WebCore::NativeImage&, const WebCore::IntSize&, WebCore::GraphicsContext&) final;
    WebGPUIdentifier backing() const { return m_backing; }

private:
    friend class WebGPU::DowncastConvertToBackingContext;

    RemoteGPUProxy(WebGPU::ConvertToBackingContext&, SerialFunctionDispatcher&);
    void initializeIPC(Ref<IPC::StreamClientConnection>&&, RenderingBackendIdentifier, IPC::StreamServerConnection::Handle&&);

    RemoteGPUProxy(const RemoteGPUProxy&) = delete;
    RemoteGPUProxy(RemoteGPUProxy&&) = delete;
    RemoteGPUProxy& operator=(const RemoteGPUProxy&) = delete;
    RemoteGPUProxy& operator=(RemoteGPUProxy&&) = delete;

    // IPC::Connection::Client
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    void didClose(IPC::Connection&) final;
    void didReceiveInvalidMessage(IPC::Connection&, IPC::MessageName, int32_t indexOfObjectFailingDecoding) final { }

    // Messages to be received.
    void wasCreated(bool didSucceed, IPC::Semaphore&& wakeUpSemaphore, IPC::Semaphore&& clientWaitSemaphore);

    void waitUntilInitialized();

    template<typename T>
    WARN_UNUSED_RETURN IPC::Error send(T&& message)
    {
        return root().protectedStreamClientConnection()->send(std::forward<T>(message), backing());
    }
    template<typename T>
    WARN_UNUSED_RETURN IPC::Connection::SendSyncResult<T> sendSync(T&& message)
    {
        return root().protectedStreamClientConnection()->sendSync(std::forward<T>(message), backing());
    }

    void requestAdapter(const WebCore::WebGPU::RequestAdapterOptions&, CompletionHandler<void(RefPtr<WebCore::WebGPU::Adapter>&&)>&&) final;

    RefPtr<WebCore::WebGPU::PresentationContext> createPresentationContext(const WebCore::WebGPU::PresentationContextDescriptor&) final;

    RefPtr<WebCore::WebGPU::CompositorIntegration> createCompositorIntegration() final;
    bool isValid(const WebCore::WebGPU::CompositorIntegration&) const final;
    bool isValid(const WebCore::WebGPU::Buffer&) const final;
    bool isValid(const WebCore::WebGPU::Adapter&) const final;
    bool isValid(const WebCore::WebGPU::BindGroup&) const final;
    bool isValid(const WebCore::WebGPU::BindGroupLayout&) const final;
    bool isValid(const WebCore::WebGPU::CommandBuffer&) const final;
    bool isValid(const WebCore::WebGPU::CommandEncoder&) const final;
    bool isValid(const WebCore::WebGPU::ComputePassEncoder&) const final;
    bool isValid(const WebCore::WebGPU::ComputePipeline&) const final;
    bool isValid(const WebCore::WebGPU::Device&) const final;
    bool isValid(const WebCore::WebGPU::ExternalTexture&) const final;
    bool isValid(const WebCore::WebGPU::PipelineLayout&) const final;
    bool isValid(const WebCore::WebGPU::PresentationContext&) const final;
    bool isValid(const WebCore::WebGPU::QuerySet&) const final;
    bool isValid(const WebCore::WebGPU::Queue&) const final;
    bool isValid(const WebCore::WebGPU::RenderBundleEncoder&) const final;
    bool isValid(const WebCore::WebGPU::RenderBundle&) const final;
    bool isValid(const WebCore::WebGPU::RenderPassEncoder&) const final;
    bool isValid(const WebCore::WebGPU::RenderPipeline&) const final;
    bool isValid(const WebCore::WebGPU::Sampler&) const final;
    bool isValid(const WebCore::WebGPU::ShaderModule&) const final;
    bool isValid(const WebCore::WebGPU::Texture&) const final;
    bool isValid(const WebCore::WebGPU::TextureView&) const final;
    bool isValid(const WebCore::WebGPU::XRBinding&) const final;
    bool isValid(const WebCore::WebGPU::XRSubImage&) const final;
    bool isValid(const WebCore::WebGPU::XRProjectionLayer&) const final;
    bool isValid(const WebCore::WebGPU::XRView&) const final;

    void abandonGPUProcess();
    void disconnectGpuProcessIfNeeded();

    // SerialFunctionDispatcher
    void dispatch(Function<void()>&&) final;
    bool isCurrent() const final;

    RefPtr<IPC::StreamClientConnection> protectedStreamConnection() const { return m_streamConnection; }
    Ref<WebGPU::ConvertToBackingContext> protectedConvertToBackingContext() const;

    Ref<WebGPU::ConvertToBackingContext> m_convertToBackingContext;
    ThreadSafeWeakPtr<SerialFunctionDispatcher> m_dispatcher;
    WeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    RefPtr<IPC::StreamClientConnection> m_streamConnection;
    WebGPUIdentifier m_backing { WebGPUIdentifier::generate() };
    bool m_didInitialize { false };
    bool m_lost { false };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
