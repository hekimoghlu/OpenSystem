/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "GPUProcessConnection.h"
#include "RemoteVideoFrameIdentifier.h"
#include "RemoteVideoFrameProxyProperties.h"
#include <WebCore/VideoFrame.h>
#include <wtf/ArgumentCoder.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
#if PLATFORM(COCOA)
class VideoFrameCV;
#endif
}

namespace WebKit {

class GPUProcessConnection;
class RemoteVideoFrameObjectHeapProxy;

// A WebCore::VideoFrame class that points to a concrete WebCore::VideoFrame instance
// in another process, GPU process.
class RemoteVideoFrameProxy final : public WebCore::VideoFrame {
    WTF_MAKE_TZONE_ALLOCATED(RemoteVideoFrameProxy);
    WTF_MAKE_NONCOPYABLE(RemoteVideoFrameProxy);
public:
    using Properties = RemoteVideoFrameProxyProperties;
    static Properties properties(WebKit::RemoteVideoFrameReference, const WebCore::VideoFrame&);

    static Ref<RemoteVideoFrameProxy> create(IPC::Connection&, RemoteVideoFrameObjectHeapProxy&, Properties&&);

    // Called by the end-points that capture creation messages that are sent from GPUP but
    // whose destinations were released in WP before message was processed.
    static void releaseUnused(IPC::Connection&, Properties&&);

    ~RemoteVideoFrameProxy() final;

    RemoteVideoFrameIdentifier identifier() const;
    RemoteVideoFrameReadReference newReadReference() const;

    WebCore::IntSize size() const { return m_size; }

    // WebCore::VideoFrame overrides.
    WebCore::IntSize presentationSize() const final { return m_size; }
    uint32_t pixelFormat() const final;
    bool isRemoteProxy() const final { return true; }
#if PLATFORM(COCOA)
    CVPixelBufferRef pixelBuffer() const final;
#endif

private:
    RemoteVideoFrameProxy(IPC::Connection&, RemoteVideoFrameObjectHeapProxy&, Properties&&);

    enum CloneConstructor { cloneConstructor };
    RemoteVideoFrameProxy(CloneConstructor, RemoteVideoFrameProxy&);

    Ref<VideoFrame> clone() final;

    static inline Seconds defaultTimeout = 10_s;

    const RefPtr<RemoteVideoFrameProxy> m_baseVideoFrame;
    const RefPtr<IPC::Connection> m_connection;
    std::optional<RemoteVideoFrameReferenceTracker> m_referenceTracker;
    const WebCore::IntSize m_size;
    uint32_t m_pixelFormat { 0 };
    // FIXME: Remove this.
    mutable RefPtr<RemoteVideoFrameObjectHeapProxy> m_videoFrameObjectHeapProxy;
#if PLATFORM(COCOA)
    mutable Lock m_pixelBufferLock;
    mutable RetainPtr<CVPixelBufferRef> m_pixelBuffer;
#endif
};

TextStream& operator<<(TextStream&, const RemoteVideoFrameProxy::Properties&);

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::RemoteVideoFrameProxy)
    static bool isType(const WebCore::VideoFrame& videoFrame) { return videoFrame.isRemoteProxy(); }
SPECIALIZE_TYPE_TRAITS_END()

#endif
