/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
#include "WebWorkerClient.h"

#include "ImageBufferShareableBitmapBackend.h"
#include "RemoteGPUProxy.h"
#include "RemoteImageBufferProxy.h"
#include "RemoteRenderingBackendProxy.h"
#include "WebGPUDowncastConvertToBackingContext.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/Page.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WEBGL) && ENABLE(GPU_PROCESS)
#include "RemoteGraphicsContextGLProxy.h"
#endif

#if ENABLE(WEBGL)
#include <WebCore/GraphicsContextGL.h>
#endif

namespace WebKit {
using namespace WebCore;

#if ENABLE(GPU_PROCESS)
class GPUProcessWebWorkerClient final : public WebWorkerClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(GPUProcessWebWorkerClient);
public:
    using WebWorkerClient::WebWorkerClient;
    UniqueRef<WorkerClient> createNestedWorkerClient(SerialFunctionDispatcher&) final;
    RefPtr<WebCore::ImageBuffer> sinkIntoImageBuffer(std::unique_ptr<WebCore::SerializedImageBuffer>) final;
    RefPtr<WebCore::ImageBuffer> createImageBuffer(const WebCore::FloatSize&, WebCore::RenderingMode, WebCore::RenderingPurpose, float resolutionScale, const WebCore::DestinationColorSpace&, WebCore::ImageBufferPixelFormat) const final;
#if ENABLE(WEBGL)
    RefPtr<WebCore::GraphicsContextGL> createGraphicsContextGL(const WebCore::GraphicsContextGLAttributes&) const final;
#endif
#if HAVE(WEBGPU_IMPLEMENTATION)
    RefPtr<WebCore::WebGPU::GPU> createGPUForWebGPU() const override;
#endif
private:
    RemoteRenderingBackendProxy& ensureRenderingBackend() const;
    Ref<RemoteRenderingBackendProxy> ensureProtectedRenderingBackend() const { return ensureRenderingBackend(); }

    mutable RefPtr<RemoteRenderingBackendProxy> m_remoteRenderingBackendProxy;
};


UniqueRef<WorkerClient> GPUProcessWebWorkerClient::createNestedWorkerClient(SerialFunctionDispatcher& dispatcher)
{
    assertIsCurrent(*this->dispatcher());
    return UniqueRef<WorkerClient> { *new GPUProcessWebWorkerClient { dispatcher, m_displayID } };
}

RemoteRenderingBackendProxy& GPUProcessWebWorkerClient::ensureRenderingBackend() const
{
    RefPtr dispatcher = this->dispatcher();
    RELEASE_ASSERT(dispatcher);
    assertIsCurrent(*dispatcher);
    if (!m_remoteRenderingBackendProxy)
        m_remoteRenderingBackendProxy = RemoteRenderingBackendProxy::create(*dispatcher);
    return *m_remoteRenderingBackendProxy;
}

RefPtr<ImageBuffer> GPUProcessWebWorkerClient::sinkIntoImageBuffer(std::unique_ptr<SerializedImageBuffer> imageBuffer)
{
    RefPtr dispatcher = this->dispatcher();
    if (!dispatcher)
        return nullptr;
    if (is<RemoteSerializedImageBufferProxy>(imageBuffer)) {
        auto remote = std::unique_ptr<RemoteSerializedImageBufferProxy>(static_cast<RemoteSerializedImageBufferProxy*>(imageBuffer.release()));
        return RemoteSerializedImageBufferProxy::sinkIntoImageBuffer(WTFMove(remote), ensureProtectedRenderingBackend());
    }
    return WebWorkerClient::sinkIntoImageBuffer(WTFMove(imageBuffer));
}

RefPtr<ImageBuffer> GPUProcessWebWorkerClient::createImageBuffer(const FloatSize& size, RenderingMode renderingMode, RenderingPurpose purpose, float resolutionScale, const DestinationColorSpace& colorSpace, ImageBufferPixelFormat pixelFormat) const
{
    if (RefPtr dispatcher = this->dispatcher())
        assertIsCurrent(*dispatcher);
    if (WebProcess::singleton().shouldUseRemoteRenderingFor(purpose))
        return ensureProtectedRenderingBackend()->createImageBuffer(size, renderingMode, purpose, resolutionScale, colorSpace, pixelFormat);
    return nullptr;
}

#if ENABLE(WEBGL)
RefPtr<GraphicsContextGL> GPUProcessWebWorkerClient::createGraphicsContextGL(const GraphicsContextGLAttributes& attributes) const
{
    RefPtr dispatcher = this->dispatcher();
    if (!dispatcher)
        return nullptr;
    assertIsCurrent(*dispatcher);
    if (WebProcess::singleton().shouldUseRemoteRenderingForWebGL())
        return RemoteGraphicsContextGLProxy::create(attributes, ensureProtectedRenderingBackend(), *dispatcher);
    return WebWorkerClient::createGraphicsContextGL(attributes);
}
#endif

#if HAVE(WEBGPU_IMPLEMENTATION)
RefPtr<WebCore::WebGPU::GPU> GPUProcessWebWorkerClient::createGPUForWebGPU() const
{
    RefPtr dispatcher = this->dispatcher();
    if (!dispatcher)
        return nullptr;
    assertIsCurrent(*dispatcher);
    return RemoteGPUProxy::create(WebGPU::DowncastConvertToBackingContext::create(), ensureProtectedRenderingBackend(), *dispatcher);
}
#endif

#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebWorkerClient);

UniqueRef<WebWorkerClient> WebWorkerClient::create(Page& page, SerialFunctionDispatcher& dispatcher)
{
    ASSERT(isMainRunLoop());
#if ENABLE(GPU_PROCESS)
    return UniqueRef<GPUProcessWebWorkerClient> { *new GPUProcessWebWorkerClient { dispatcher, page.displayID() } };
#else
    return UniqueRef<WebWorkerClient> { *new WebWorkerClient { dispatcher, page.displayID() } };
#endif
}

WebWorkerClient::WebWorkerClient(SerialFunctionDispatcher& dispatcher, WebCore::PlatformDisplayID displayID)
    : m_dispatcher(dispatcher)
    , m_displayID(displayID)
{
}

WebWorkerClient::~WebWorkerClient() = default;

UniqueRef<WorkerClient> WebWorkerClient::createNestedWorkerClient(SerialFunctionDispatcher& dispatcher)
{
    assertIsCurrent(*this->dispatcher().get());
    return UniqueRef<WorkerClient> { *new WebWorkerClient { dispatcher, m_displayID } };
}

PlatformDisplayID WebWorkerClient::displayID() const
{
    assertIsCurrent(*dispatcher().get());
    return m_displayID;
}

RefPtr<ImageBuffer> WebWorkerClient::sinkIntoImageBuffer(std::unique_ptr<SerializedImageBuffer> imageBuffer)
{
    assertIsCurrent(*dispatcher().get());
    return SerializedImageBuffer::sinkIntoImageBuffer(WTFMove(imageBuffer));
}

RefPtr<ImageBuffer> WebWorkerClient::createImageBuffer(const FloatSize& size, RenderingMode renderingMode, RenderingPurpose purpose, float resolutionScale, const DestinationColorSpace& colorSpace, ImageBufferPixelFormat pixelFormat) const
{
    assertIsCurrent(*dispatcher().get());
    return nullptr;
}

#if ENABLE(WEBGL)
RefPtr<GraphicsContextGL> WebWorkerClient::createGraphicsContextGL(const GraphicsContextGLAttributes& attributes) const
{
    assertIsCurrent(*dispatcher().get());
    return WebCore::createWebProcessGraphicsContextGL(attributes);
}
#endif

#if HAVE(WEBGPU_IMPLEMENTATION)
RefPtr<WebCore::WebGPU::GPU> WebWorkerClient::createGPUForWebGPU() const
{
    return nullptr;
}
#endif

}
