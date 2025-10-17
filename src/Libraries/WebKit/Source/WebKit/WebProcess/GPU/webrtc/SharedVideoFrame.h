/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA) && ENABLE(VIDEO)

#include "IPCSemaphore.h"
#include "RemoteVideoFrameIdentifier.h"
#include <WebCore/IntSize.h>
#include <WebCore/ProcessIdentity.h>
#include <WebCore/SharedMemory.h>
#include <wtf/MediaTime.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

typedef struct __CVBuffer* CVPixelBufferRef;
typedef struct __CVPixelBufferPool* CVPixelBufferPoolRef;

namespace WebCore {
class SharedVideoFrameInfo;
class VideoFrame;
enum class VideoFrameRotation : uint16_t;
}

namespace webrtc {
class VideoFrame;
class VideoFrameBuffer;
}

namespace WebKit {

class RemoteVideoFrameObjectHeap;

struct SharedVideoFrame {
    MediaTime time;
    bool mirrored { false };
    WebCore::VideoFrameRotation rotation { };
    using Buffer = std::variant<std::nullptr_t, RemoteVideoFrameReadReference, MachSendRight, WebCore::IntSize>;
    Buffer buffer;
};

class SharedVideoFrameWriter {
    WTF_MAKE_TZONE_ALLOCATED(SharedVideoFrameWriter);
public:
    SharedVideoFrameWriter();

    std::optional<SharedVideoFrame> write(const WebCore::VideoFrame&, const Function<void(IPC::Semaphore&)>&, const Function<void(WebCore::SharedMemory::Handle&&)>&);
    std::optional<SharedVideoFrame::Buffer> writeBuffer(CVPixelBufferRef, const Function<void(IPC::Semaphore&)>&, const Function<void(WebCore::SharedMemory::Handle&&)>&, bool canSendIOSurface = true);
#if USE(LIBWEBRTC)
    std::optional<SharedVideoFrame::Buffer> writeBuffer(const webrtc::VideoFrame&, const Function<void(IPC::Semaphore&)>&, const Function<void(WebCore::SharedMemory::Handle&&)>&);
#endif
    std::optional<SharedVideoFrame::Buffer> writeBuffer(const WebCore::VideoFrame&, const Function<void(IPC::Semaphore&)>&, const Function<void(WebCore::SharedMemory::Handle&&)>&);

    void disable();
    bool isDisabled() const { return m_isDisabled; }

private:
    static constexpr Seconds defaultTimeout = 3_s;

    bool wait(const Function<void(IPC::Semaphore&)>&);
    bool allocateStorage(size_t, const Function<void(WebCore::SharedMemory::Handle&&)>&);
    bool prepareWriting(const WebCore::SharedVideoFrameInfo&, const Function<void(IPC::Semaphore&)>&, const Function<void(WebCore::SharedMemory::Handle&&)>&);

#if USE(LIBWEBRTC)
    std::optional<SharedVideoFrame::Buffer> writeBuffer(webrtc::VideoFrameBuffer&, const Function<void(IPC::Semaphore&)>&, const Function<void(WebCore::SharedMemory::Handle&&)>&);
#endif
    void signalInCaseOfError();

    UniqueRef<IPC::Semaphore> m_semaphore;
    RefPtr<WebCore::SharedMemory> m_storage;
    bool m_isSemaphoreInUse { false };
    bool m_isDisabled { false };
    bool m_shouldSignalInCaseOfError { false };
};

class SharedVideoFrameReader {
    WTF_MAKE_TZONE_ALLOCATED(SharedVideoFrameReader);
public:
    ~SharedVideoFrameReader();

    enum class UseIOSurfaceBufferPool : bool { No, Yes };
    explicit SharedVideoFrameReader(RefPtr<RemoteVideoFrameObjectHeap>&&, const WebCore::ProcessIdentity& = { }, UseIOSurfaceBufferPool = UseIOSurfaceBufferPool::Yes);
    SharedVideoFrameReader();

    void setSemaphore(IPC::Semaphore&& semaphore) { m_semaphore = WTFMove(semaphore); }
    bool setSharedMemory(WebCore::SharedMemory::Handle&&);

    RefPtr<WebCore::VideoFrame> read(SharedVideoFrame&&);
    RetainPtr<CVPixelBufferRef> readBuffer(SharedVideoFrame::Buffer&&);

private:
    CVPixelBufferPoolRef pixelBufferPool(const WebCore::SharedVideoFrameInfo&);
    RetainPtr<CVPixelBufferRef> readBufferFromSharedMemory();

    RefPtr<RemoteVideoFrameObjectHeap> m_objectHeap;
    WebCore::ProcessIdentity m_resourceOwner;
    UseIOSurfaceBufferPool m_useIOSurfaceBufferPool { UseIOSurfaceBufferPool::No };
    std::optional<IPC::Semaphore> m_semaphore;
    RefPtr<WebCore::SharedMemory> m_storage;

    RetainPtr<CVPixelBufferPoolRef> m_bufferPool;
    OSType m_bufferPoolType { 0 };
    uint32_t m_bufferPoolWidth { 0 };
    uint32_t m_bufferPoolHeight { 0 };
    WebCore::IntSize m_blackFrameSize;
    RetainPtr<CVPixelBufferRef> m_blackFrame;
};

}

#endif
