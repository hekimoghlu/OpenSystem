/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 22, 2023.
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
#include "RemoteGPUProxy.h"

#if ENABLE(GPU_PROCESS)

#include "GPUConnectionToWebProcessMessages.h"
#include "GPUProcessConnection.h"
#include "RemoteAdapterProxy.h"
#include "RemoteCompositorIntegrationProxy.h"
#include "RemoteGPU.h"
#include "RemoteGPUMessages.h"
#include "RemoteGPUProxyMessages.h"
#include "RemotePresentationContextProxy.h"
#include "RemoteRenderingBackendProxy.h"
#include "WebGPUConvertToBackingContext.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/WebGPUPresentationContextDescriptor.h>
#include <WebCore/WebGPUSupportedFeatures.h>
#include <WebCore/WebGPUSupportedLimits.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteGPUProxy);

RefPtr<RemoteGPUProxy> RemoteGPUProxy::create(WebGPU::ConvertToBackingContext& convertToBackingContext, WebPage& page)
{
    return RemoteGPUProxy::create(convertToBackingContext, page.ensureProtectedRemoteRenderingBackendProxy(), RunLoop::protectedMain().get());
}

RefPtr<RemoteGPUProxy> RemoteGPUProxy::create(WebGPU::ConvertToBackingContext& convertToBackingContext, RemoteRenderingBackendProxy& renderingBackend, SerialFunctionDispatcher& dispatcher)
{
    constexpr size_t connectionBufferSizeLog2 = 21;
    auto connectionPair = IPC::StreamClientConnection::create(connectionBufferSizeLog2, WebProcess::singleton().gpuProcessTimeoutDuration());
    if (!connectionPair)
        return nullptr;
    auto [clientConnection, serverConnectionHandle] = WTFMove(*connectionPair);
    Ref instance = adoptRef(*new RemoteGPUProxy(convertToBackingContext, dispatcher));
    instance->initializeIPC(WTFMove(clientConnection), renderingBackend.ensureBackendCreated(), WTFMove(serverConnectionHandle));
    // TODO: We must wait until initialized, because at the moment we cannot receive IPC messages
    // during wait while in synchronous stream send. Should be fixed as part of https://bugs.webkit.org/show_bug.cgi?id=217211.
    instance->waitUntilInitialized();
    return instance;
}


RemoteGPUProxy::RemoteGPUProxy(WebGPU::ConvertToBackingContext& convertToBackingContext, SerialFunctionDispatcher& dispatcher)
    : m_convertToBackingContext(convertToBackingContext)
    , m_dispatcher(dispatcher)
{
}

RemoteGPUProxy::~RemoteGPUProxy()
{
    disconnectGpuProcessIfNeeded();
}

void RemoteGPUProxy::initializeIPC(Ref<IPC::StreamClientConnection>&& streamConnection, RenderingBackendIdentifier renderingBackend, IPC::StreamServerConnection::Handle&& serverHandle)
{
    m_streamConnection = WTFMove(streamConnection);
    protectedStreamConnection()->open(*this, *this);
    callOnMainRunLoopAndWait([&]() {
        Ref gpuProcessConnection = WebProcess::singleton().ensureGPUProcessConnection();
        gpuProcessConnection->createGPU(m_backing, renderingBackend, WTFMove(serverHandle));
        m_gpuProcessConnection = gpuProcessConnection.get();
    });
}

void RemoteGPUProxy::disconnectGpuProcessIfNeeded()
{
    if (m_lost)
        return;
    protectedStreamConnection()->invalidate();
    // FIXME: deallocate m_streamConnection once the children work without the connection.
    ensureOnMainRunLoop([identifier = m_backing, weakGPUProcessConnection = WTFMove(m_gpuProcessConnection)]() {
        RefPtr gpuProcessConnection = weakGPUProcessConnection.get();
        if (!gpuProcessConnection)
            return;
        gpuProcessConnection->releaseGPU(identifier);
    });
}

void RemoteGPUProxy::didClose(IPC::Connection&)
{
    ASSERT(m_streamConnection);
    abandonGPUProcess();
}

void RemoteGPUProxy::abandonGPUProcess()
{
    protectedStreamConnection()->invalidate();
    m_lost = true;
}

void RemoteGPUProxy::dispatch(Function<void()>&& function)
{
    if (RefPtr dispatcher = m_dispatcher.get())
        dispatcher->dispatch(WTFMove(function));
}

bool RemoteGPUProxy::isCurrent() const
{
    RefPtr dispatcher = m_dispatcher.get();
    return dispatcher && dispatcher->isCurrent();
}

void RemoteGPUProxy::wasCreated(bool didSucceed, IPC::Semaphore&& wakeUpSemaphore, IPC::Semaphore&& clientWaitSemaphore)
{
    ASSERT(!m_didInitialize);
    m_didInitialize = true;
    if (didSucceed)
        protectedStreamConnection()->setSemaphores(WTFMove(wakeUpSemaphore), WTFMove(clientWaitSemaphore));
    else
        abandonGPUProcess();
}

void RemoteGPUProxy::waitUntilInitialized()
{
    if (m_didInitialize)
        return;
    if (protectedStreamConnection()->waitForAndDispatchImmediately<Messages::RemoteGPUProxy::WasCreated>(m_backing) == IPC::Error::NoError)
        return;
    abandonGPUProcess();
}

void RemoteGPUProxy::requestAdapter(const WebCore::WebGPU::RequestAdapterOptions& options, CompletionHandler<void(RefPtr<WebCore::WebGPU::Adapter>&&)>&& callback)
{
    if (m_lost) {
        callback(nullptr);
        return;
    }

    Ref convertToBackingContext = m_convertToBackingContext;
    auto convertedOptions = convertToBackingContext->convertToBacking(options);
    ASSERT(convertedOptions);
    if (!convertedOptions) {
        callback(nullptr);
        return;
    }

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = sendSync(Messages::RemoteGPU::RequestAdapter(*convertedOptions, identifier));
    if (!sendResult.succeeded()) {
        abandonGPUProcess();
        callback(nullptr);
        return;
    }
    auto [response] = sendResult.takeReply();
    if (!response) {
        callback(nullptr);
        return;
    }

    Ref resultSupportedFeatures = WebCore::WebGPU::SupportedFeatures::create(WTFMove(response->features.features));
    Ref resultSupportedLimits = WebCore::WebGPU::SupportedLimits::create(
        response->limits.maxTextureDimension1D,
        response->limits.maxTextureDimension2D,
        response->limits.maxTextureDimension3D,
        response->limits.maxTextureArrayLayers,
        response->limits.maxBindGroups,
        response->limits.maxBindGroupsPlusVertexBuffers,
        response->limits.maxBindingsPerBindGroup,
        response->limits.maxDynamicUniformBuffersPerPipelineLayout,
        response->limits.maxDynamicStorageBuffersPerPipelineLayout,
        response->limits.maxSampledTexturesPerShaderStage,
        response->limits.maxSamplersPerShaderStage,
        response->limits.maxStorageBuffersPerShaderStage,
        response->limits.maxStorageTexturesPerShaderStage,
        response->limits.maxUniformBuffersPerShaderStage,
        response->limits.maxUniformBufferBindingSize,
        response->limits.maxStorageBufferBindingSize,
        response->limits.minUniformBufferOffsetAlignment,
        response->limits.minStorageBufferOffsetAlignment,
        response->limits.maxVertexBuffers,
        response->limits.maxBufferSize,
        response->limits.maxVertexAttributes,
        response->limits.maxVertexBufferArrayStride,
        response->limits.maxInterStageShaderComponents,
        response->limits.maxInterStageShaderVariables,
        response->limits.maxColorAttachments,
        response->limits.maxColorAttachmentBytesPerSample,
        response->limits.maxComputeWorkgroupStorageSize,
        response->limits.maxComputeInvocationsPerWorkgroup,
        response->limits.maxComputeWorkgroupSizeX,
        response->limits.maxComputeWorkgroupSizeY,
        response->limits.maxComputeWorkgroupSizeZ,
        response->limits.maxComputeWorkgroupsPerDimension,
        response->limits.maxStorageBuffersInFragmentStage,
        response->limits.maxStorageTexturesInFragmentStage
    );
    callback(WebGPU::RemoteAdapterProxy::create(WTFMove(response->name), WTFMove(resultSupportedFeatures), WTFMove(resultSupportedLimits), response->isFallbackAdapter, options.xrCompatible, *this, convertToBackingContext, identifier));
}

RefPtr<WebCore::WebGPU::PresentationContext> RemoteGPUProxy::createPresentationContext(const WebCore::WebGPU::PresentationContextDescriptor& descriptor)
{
    // FIXME: Should we be consulting m_lost?

    // FIXME: This is super yucky. We should solve this a better way. (For both WK1 and WK2.)
    // Maybe PresentationContext needs a present() function?
    Ref convertToBackingContext = m_convertToBackingContext;
    Ref compositorIntegration = const_cast<WebGPU::RemoteCompositorIntegrationProxy&>(convertToBackingContext->convertToRawBacking(Ref { descriptor.compositorIntegration }.get()));

    auto convertedDescriptor = convertToBackingContext->convertToBacking(descriptor);
    if (!convertedDescriptor)
        return nullptr;

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteGPU::CreatePresentationContext(*convertedDescriptor, identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    Ref result = WebGPU::RemotePresentationContextProxy::create(*this, convertToBackingContext, identifier);
    compositorIntegration->setPresentationContext(result);
    return result;
}

RefPtr<WebCore::WebGPU::CompositorIntegration> RemoteGPUProxy::createCompositorIntegration()
{
    // FIXME: Should we be consulting m_lost?

    auto identifier = WebGPUIdentifier::generate();
    auto sendResult = send(Messages::RemoteGPU::CreateCompositorIntegration(identifier));
    if (sendResult != IPC::Error::NoError)
        return nullptr;

    return WebGPU::RemoteCompositorIntegrationProxy::create(*this, protectedConvertToBackingContext(), identifier);
}

void RemoteGPUProxy::paintToCanvas(WebCore::NativeImage&, const WebCore::IntSize&, WebCore::GraphicsContext&)
{
    ASSERT_NOT_REACHED();
}

Ref<WebGPU::ConvertToBackingContext> RemoteGPUProxy::protectedConvertToBackingContext() const
{
    return m_convertToBackingContext;
}

bool RemoteGPUProxy::isValid(const WebCore::WebGPU::CompositorIntegration&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::Buffer&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::Adapter&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::BindGroup&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::BindGroupLayout&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::CommandBuffer&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::CommandEncoder&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::ComputePassEncoder&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::ComputePipeline&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::Device&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::ExternalTexture&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::PipelineLayout&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::PresentationContext&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::QuerySet&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::Queue&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::RenderBundleEncoder&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::RenderBundle&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::RenderPassEncoder&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::RenderPipeline&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::Sampler&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::ShaderModule&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::Texture&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::TextureView&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::XRBinding&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::XRSubImage&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::XRProjectionLayer&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}
bool RemoteGPUProxy::isValid(const WebCore::WebGPU::XRView&) const
{
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
