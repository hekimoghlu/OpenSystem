/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

#include "BufferIdentifierSet.h"
#include "MarkSurfacesAsVolatileRequestIdentifier.h"
#include "PrepareBackingStoreBuffersData.h"
#include "RemoteDisplayListRecorderProxy.h"
#include "RemoteImageBufferSetIdentifier.h"
#include "RenderingUpdateID.h"
#include "WorkQueueMessageReceiver.h"
#include <wtf/Identified.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(GPU_PROCESS)

namespace IPC {
class Connection;
class Decoder;
class StreamClientConnection;
}

namespace WebKit {

class RemoteImageBufferSetProxyFlushFence;
struct BufferSetBackendHandle;

// FIXME: We should have a generic 'ImageBufferSet' class that contains
// the code that isn't specific to being remote, and this helper belongs
// there.
class ThreadSafeImageBufferSetFlusher {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ThreadSafeImageBufferSetFlusher);
    WTF_MAKE_NONCOPYABLE(ThreadSafeImageBufferSetFlusher);
public:
    enum class FlushType {
        BackendHandlesOnly,
        BackendHandlesAndDrawing,
    };

    ThreadSafeImageBufferSetFlusher() = default;
    virtual ~ThreadSafeImageBufferSetFlusher() = default;
    // Returns true if flush succeeded, false if it failed.
    virtual bool flushAndCollectHandles(HashMap<RemoteImageBufferSetIdentifier, std::unique_ptr<BufferSetBackendHandle>>&) = 0;
};

// A RemoteImageBufferSet is an ImageBufferSet, where the actual ImageBuffers are owned by the GPU process.
// To draw a frame, the consumer allocates a new RemoteDisplayListRecorderProxy and
// asks the RemoteImageBufferSet set to map it to an appropriate new front
// buffer (either by picking one of the back buffers, or by allocating a new
// one). It then copies across the pixels from the previous front buffer,
// clips to the dirty region and clears that region.
// Usage is done through RemoteRenderingBackendProxy::prepareImageBufferSetsForDisplay,
// so that a Vector of RemoteImageBufferSets can be used with a single
// IPC call.
// FIXME: It would be nice if this could actually be a subclass of ImageBufferSet, but
// probably can't while it uses batching for prepare and volatility.
class RemoteImageBufferSetProxy : public IPC::WorkQueueMessageReceiver, public Identified<RemoteImageBufferSetIdentifier> {
public:
    RemoteImageBufferSetProxy(RemoteRenderingBackendProxy&);
    ~RemoteImageBufferSetProxy();

    OptionSet<BufferInSetType> requestedVolatility() { return m_requestedVolatility; }
    OptionSet<BufferInSetType> confirmedVolatility() { return m_confirmedVolatility; }
    void clearVolatility();
    void addRequestedVolatility(OptionSet<BufferInSetType> request);
    void setConfirmedVolatility(OptionSet<BufferInSetType> types);

#if PLATFORM(COCOA)
    void didPrepareForDisplay(ImageBufferSetPrepareBufferForDisplayOutputData, RenderingUpdateID);
#endif

    WebCore::GraphicsContext& context();
    bool hasContext() const { return !!m_displayListRecorder; }

    WebCore::RenderingResourceIdentifier displayListResourceIdentifier() const { return m_displayListIdentifier; }

    std::unique_ptr<ThreadSafeImageBufferSetFlusher> flushFrontBufferAsync(ThreadSafeImageBufferSetFlusher::FlushType);

    void setConfiguration(WebCore::FloatSize, float, const WebCore::DestinationColorSpace&, WebCore::ContentsFormat, WebCore::ImageBufferPixelFormat, WebCore::RenderingMode, WebCore::RenderingPurpose);
    void willPrepareForDisplay();
    void remoteBufferSetWasDestroyed();

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    std::optional<WebCore::DynamicContentScalingDisplayList> dynamicContentScalingDisplayList();
#endif

    unsigned generation() const { return m_generation; }

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

    void close();

private:
    template<typename T> auto send(T&& message);
    template<typename T> auto sendSync(T&& message);
    RefPtr<IPC::StreamClientConnection> connection() const;
    void didBecomeUnresponsive() const;

    WeakPtr<RemoteRenderingBackendProxy> m_remoteRenderingBackendProxy;

    WebCore::RenderingResourceIdentifier m_displayListIdentifier;
    std::unique_ptr<RemoteDisplayListRecorderProxy> m_displayListRecorder;

    OptionSet<BufferInSetType> m_requestedVolatility;
    OptionSet<BufferInSetType> m_confirmedVolatility;

    WebCore::FloatSize m_size;
    float m_scale { 1.0f };
    WebCore::DestinationColorSpace m_colorSpace { WebCore::DestinationColorSpace::SRGB() };
    WebCore::ContentsFormat m_contentsFormat { WebCore::ContentsFormat::RGBA8 };
    WebCore::ImageBufferPixelFormat m_pixelFormat;
    WebCore::RenderingMode m_renderingMode { WebCore::RenderingMode::Unaccelerated };
    WebCore::RenderingPurpose m_renderingPurpose { WebCore::RenderingPurpose::Unspecified };
    unsigned m_generation { 0 };
    bool m_remoteNeedsConfigurationUpdate { false };

    Lock m_lock;
    RefPtr<RemoteImageBufferSetProxyFlushFence> m_pendingFlush WTF_GUARDED_BY_LOCK(m_lock);
    RefPtr<IPC::StreamClientConnection> m_streamConnection  WTF_GUARDED_BY_LOCK(m_lock);
    bool m_prepareForDisplayIsPending WTF_GUARDED_BY_LOCK(m_lock) { false };
    bool m_closed WTF_GUARDED_BY_LOCK(m_lock) { false };
};

inline TextStream& operator<<(TextStream& ts, RemoteImageBufferSetProxy& bufferSet)
{
    ts << bufferSet.identifier();
    return ts;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
