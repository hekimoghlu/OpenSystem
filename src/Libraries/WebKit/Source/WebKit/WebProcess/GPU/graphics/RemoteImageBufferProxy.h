/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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

#include "ImageBufferBackendHandle.h"
#include "RemoteDisplayListRecorderProxy.h"
#include "RemoteSerializedImageBufferIdentifier.h"
#include "RenderingBackendIdentifier.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/ImageBufferBackend.h>
#include <wtf/Condition.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class Connection;
}

namespace WebKit {

class RemoteRenderingBackendProxy;

class RemoteImageBufferProxy : public WebCore::ImageBuffer {
    WTF_MAKE_TZONE_ALLOCATED(RemoteImageBufferProxy);
    friend class RemoteSerializedImageBufferProxy;
public:
    template<typename BackendType>
    static RefPtr<RemoteImageBufferProxy> create(const WebCore::FloatSize& size, float resolutionScale, const WebCore::DestinationColorSpace& colorSpace, WebCore::ImageBufferPixelFormat pixelFormat, WebCore::RenderingPurpose purpose, RemoteRenderingBackendProxy& remoteRenderingBackendProxy)
    {
        Parameters parameters { size, resolutionScale, colorSpace, pixelFormat, purpose };
        auto backendParameters = ImageBuffer::backendParameters(parameters);
        if (BackendType::calculateSafeBackendSize(backendParameters).isEmpty())
            return nullptr;
        auto info = populateBackendInfo<BackendType>(backendParameters);
        return adoptRef(new RemoteImageBufferProxy(parameters, info, remoteRenderingBackendProxy));
    }

    ~RemoteImageBufferProxy();

    WebCore::ImageBufferBackend* ensureBackend() const final;

    void clearBackend();
    void backingStoreWillChange();
    std::unique_ptr<WebCore::SerializedImageBuffer> sinkIntoSerializedImageBuffer() final;

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    // Messages
    void didCreateBackend(std::optional<ImageBufferBackendHandle>);

private:
    RemoteImageBufferProxy(Parameters, const WebCore::ImageBufferBackend::Info&, RemoteRenderingBackendProxy&, std::unique_ptr<WebCore::ImageBufferBackend>&& = nullptr, WebCore::RenderingResourceIdentifier = WebCore::RenderingResourceIdentifier::generate());

    RefPtr<WebCore::NativeImage> copyNativeImage() const final;
    RefPtr<WebCore::NativeImage> createNativeImageReference() const final;
    RefPtr<WebCore::NativeImage> sinkIntoNativeImage() final;

    RefPtr<ImageBuffer> sinkIntoBufferForDifferentThread() final;

    RefPtr<WebCore::NativeImage> filteredNativeImage(WebCore::Filter&) final;

    WebCore::GraphicsContext& context() const final;

    RefPtr<WebCore::PixelBuffer> getPixelBuffer(const WebCore::PixelBufferFormat& destinationFormat, const WebCore::IntRect& srcRect, const WebCore::ImageBufferAllocator&) const final;
    void putPixelBuffer(const WebCore::PixelBuffer&, const WebCore::IntRect& srcRect, const WebCore::IntPoint& destPoint = { }, WebCore::AlphaPremultiplication = WebCore::AlphaPremultiplication::Premultiplied) final;

    void convertToLuminanceMask() final;
    void transformToColorSpace(const WebCore::DestinationColorSpace&) final;

    void flushDrawingContext() final;
    bool flushDrawingContextAsync() final;

    void prepareForBackingStoreChange();

    void assertDispatcherIsCurrent() const;
    template<typename T> void send(T&& message);
    template<typename T> auto sendSync(T&& message);
    RefPtr<IPC::StreamClientConnection> connection() const;
    void didBecomeUnresponsive() const;

    WeakPtr<RemoteRenderingBackendProxy> m_remoteRenderingBackendProxy;
    RemoteDisplayListRecorderProxy m_remoteDisplayList;
    bool m_needsFlush { true };
};

class RemoteSerializedImageBufferProxy : public WebCore::SerializedImageBuffer {
    WTF_MAKE_TZONE_ALLOCATED(RemoteSerializedImageBufferProxy);
    friend class RemoteRenderingBackendProxy;
public:
    ~RemoteSerializedImageBufferProxy();

    static RefPtr<WebCore::ImageBuffer> sinkIntoImageBuffer(std::unique_ptr<RemoteSerializedImageBufferProxy>, RemoteRenderingBackendProxy&);

    WebCore::RenderingResourceIdentifier renderingResourceIdentifier() { return m_renderingResourceIdentifier; }

    RemoteSerializedImageBufferProxy(WebCore::ImageBuffer::Parameters, const WebCore::ImageBufferBackend::Info&, const WebCore::RenderingResourceIdentifier&, RemoteRenderingBackendProxy&);

    size_t memoryCost() const final
    {
        return m_info.memoryCost;
    }

private:
    RefPtr<WebCore::ImageBuffer> sinkIntoImageBuffer() final
    {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    bool isRemoteSerializedImageBufferProxy() const final { return true; }

    WebCore::ImageBuffer::Parameters m_parameters;
    WebCore::ImageBufferBackend::Info m_info;
    WebCore::RenderingResourceIdentifier m_renderingResourceIdentifier;
    RefPtr<IPC::Connection> m_connection;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteSerializedImageBufferProxy)
    static bool isType(const WebCore::SerializedImageBuffer& buffer) { return buffer.isRemoteSerializedImageBufferProxy(); }
SPECIALIZE_TYPE_TRAITS_END()


#endif // ENABLE(GPU_PROCESS)
