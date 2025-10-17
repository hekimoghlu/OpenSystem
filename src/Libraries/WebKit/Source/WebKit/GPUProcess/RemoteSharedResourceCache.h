/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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

#include "MessageReceiver.h"
#include "RemoteSerializedImageBufferIdentifier.h"
#include "ThreadSafeObjectHeap.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/ImageBufferResourceLimits.h>
#include <WebCore/ProcessIdentity.h>
#include <WebCore/RenderingResourceIdentifier.h>
#include <wtf/FastMalloc.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeRefCounted.h>

#if USE(IOSURFACE)
#include <WebCore/IOSurfacePool.h>
#endif

namespace WebKit {

class GPUConnectionToWebProcess;
// Class holding GPU process resources per Web Content process.
// Thread-safe.
class RemoteSharedResourceCache final : public ThreadSafeRefCounted<RemoteSharedResourceCache>, IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteSharedResourceCache);
public:
    static Ref<RemoteSharedResourceCache> create(GPUConnectionToWebProcess&);
    virtual ~RemoteSharedResourceCache();

    void ref() const final { ThreadSafeRefCounted::ref(); }
    void deref() const final { ThreadSafeRefCounted::deref(); }

    void addSerializedImageBuffer(WebCore::RenderingResourceIdentifier, Ref<WebCore::ImageBuffer>);
    RefPtr<WebCore::ImageBuffer> takeSerializedImageBuffer(WebCore::RenderingResourceIdentifier);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    const WebCore::ProcessIdentity& resourceOwner() const { return m_resourceOwner; }
#if HAVE(IOSURFACE)
    WebCore::IOSurfacePool& ioSurfacePool() const { return m_ioSurfacePool; }
#endif

    void didCreateImageBuffer(WebCore::RenderingPurpose, WebCore::RenderingMode);
    void didReleaseImageBuffer(WebCore::RenderingPurpose, WebCore::RenderingMode);
    bool reachedAcceleratedImageBufferLimit(WebCore::RenderingPurpose) const;
    bool reachedImageBufferForCanvasLimit() const;
    WebCore::ImageBufferResourceLimits getResourceLimitsForTesting() const;

    void lowMemoryHandler();

private:
    RemoteSharedResourceCache(GPUConnectionToWebProcess&);

    // Messages
    void releaseSerializedImageBuffer(WebCore::RenderingResourceIdentifier);

    IPC::ThreadSafeObjectHeap<RemoteSerializedImageBufferIdentifier, RefPtr<WebCore::ImageBuffer>> m_serializedImageBuffers;
    WebCore::ProcessIdentity m_resourceOwner;
#if HAVE(IOSURFACE)
    Ref<WebCore::IOSurfacePool> m_ioSurfacePool;
#endif
    std::atomic<size_t> m_acceleratedImageBufferForCanvasCount;
    std::atomic<size_t> m_imageBufferForCanvasCount;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
