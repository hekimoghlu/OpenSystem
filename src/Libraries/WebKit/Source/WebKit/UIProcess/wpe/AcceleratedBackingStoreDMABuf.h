/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 12, 2024.
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

#if ENABLE(WPE_PLATFORM)
#include "FenceMonitor.h"
#include "MessageReceiver.h"
#include "RendererBufferFormat.h"
#include <WebCore/IntSize.h>
#include <WebCore/Region.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/unix/UnixFileDescriptor.h>

typedef struct _WPEBuffer WPEBuffer;
typedef struct _WPEView WPEView;

namespace WebCore {
class Region;
class ShareableBitmapHandle;
}

namespace WTF {
class UnixFileDescriptor;
}

namespace WebKit {

class WebPageProxy;
class WebProcessProxy;

class AcceleratedBackingStoreDMABuf final : public IPC::MessageReceiver, public RefCounted<AcceleratedBackingStoreDMABuf> {
    WTF_MAKE_TZONE_ALLOCATED(AcceleratedBackingStoreDMABuf);
    WTF_MAKE_NONCOPYABLE(AcceleratedBackingStoreDMABuf);
public:
    static Ref<AcceleratedBackingStoreDMABuf> create(WebPageProxy&, WPEView*);
    ~AcceleratedBackingStoreDMABuf();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void updateSurfaceID(uint64_t);

    RendererBufferFormat bufferFormat() const;

private:
    AcceleratedBackingStoreDMABuf(WebPageProxy&, WPEView*);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    void didCreateBuffer(uint64_t id, const WebCore::IntSize&, uint32_t format, Vector<WTF::UnixFileDescriptor>&&, Vector<uint32_t>&& offsets, Vector<uint32_t>&& strides, uint64_t modifier, DMABufRendererBufferFormat::Usage);
    void didCreateBufferSHM(uint64_t id, WebCore::ShareableBitmapHandle&&);
    void didDestroyBuffer(uint64_t id);
    void frame(uint64_t bufferID, WebCore::Region&&, WTF::UnixFileDescriptor&&);
    void frameDone();
    void renderPendingBuffer();
    void bufferRendered();
    void bufferReleased(WPEBuffer*);

    WeakPtr<WebPageProxy> m_webPage;
    GRefPtr<WPEView> m_wpeView;
    FenceMonitor m_fenceMonitor;
    uint64_t m_surfaceID { 0 };
    WeakPtr<WebProcessProxy> m_legacyMainFrameProcess;
    GRefPtr<WPEBuffer> m_pendingBuffer;
    GRefPtr<WPEBuffer> m_committedBuffer;
    WebCore::Region m_pendingDamageRegion;
    HashMap<uint64_t, GRefPtr<WPEBuffer>> m_buffers;
    HashMap<WPEBuffer*, uint64_t> m_bufferIDs;
};

} // namespace WebKit

#endif // ENABLE(WPE_PLATFORM)
