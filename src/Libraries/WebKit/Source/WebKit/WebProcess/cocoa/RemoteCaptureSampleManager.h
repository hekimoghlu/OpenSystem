/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)

#include "Connection.h"
#include "IPCSemaphore.h"
#include "MessageReceiver.h"
#include "RemoteRealtimeAudioSource.h"
#include "RemoteRealtimeVideoSource.h"
#include "RemoteVideoFrameIdentifier.h"
#include "RemoteVideoFrameProxy.h"
#include "SharedCARingBuffer.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/CAAudioStreamDescription.h>
#include <WebCore/CARingBuffer.h>
#include <WebCore/WebAudioBufferList.h>
#include <wtf/CheckedRef.h>
#include <wtf/HashMap.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WorkQueue.h>

namespace WebCore {
class ImageTransferSessionVT;
enum class VideoFrameRotation : uint16_t;
}

namespace WebKit {

class RemoteVideoFrameObjectHeapProxy;
class UserMediaCaptureManager;

class RemoteCaptureSampleManager : public IPC::WorkQueueMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteCaptureSampleManager);
public:
    explicit RemoteCaptureSampleManager(UserMediaCaptureManager&);
    ~RemoteCaptureSampleManager();

    void ref() const;
    void deref() const;

    void stopListeningForIPC();

    void addSource(Ref<RemoteRealtimeAudioSource>&&);
    void addSource(Ref<RemoteRealtimeVideoSource>&&);
    void removeSource(WebCore::RealtimeMediaSourceIdentifier);

    void didUpdateSourceConnection(IPC::Connection&);
    void setVideoFrameObjectHeapProxy(RefPtr<RemoteVideoFrameObjectHeapProxy>&&);

    // IPC::WorkQueueMessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&);

private:
    // Messages
    void audioStorageChanged(WebCore::RealtimeMediaSourceIdentifier, ConsumerSharedCARingBuffer::Handle&&, const WebCore::CAAudioStreamDescription&, IPC::Semaphore&&, const MediaTime&, size_t frameSampleSize);
    void audioSamplesAvailable(WebCore::RealtimeMediaSourceIdentifier, MediaTime, uint64_t numberOfFrames);
    void videoFrameAvailable(WebCore::RealtimeMediaSourceIdentifier, RemoteVideoFrameProxy::Properties&&, WebCore::VideoFrameTimeMetadata);
    // FIXME: Will be removed once RemoteVideoFrameProxy providers are the only ones sending data.
    void videoFrameAvailableCV(WebCore::RealtimeMediaSourceIdentifier, RetainPtr<CVPixelBufferRef>&&, WebCore::VideoFrameRotation, bool mirrored, MediaTime, WebCore::VideoFrameTimeMetadata);

    void setConnection(RefPtr<IPC::Connection>&&);

    class RemoteAudio {
        WTF_MAKE_TZONE_ALLOCATED(RemoteAudio);
    public:
        explicit RemoteAudio(Ref<RemoteRealtimeAudioSource>&&);
        ~RemoteAudio();

        void setStorage(ConsumerSharedCARingBuffer::Handle&&, const WebCore::CAAudioStreamDescription&, IPC::Semaphore&&, const MediaTime&, size_t frameChunkSize);

    private:
        void stopThread();
        void startThread();

        Ref<RemoteRealtimeAudioSource> m_source;
        std::optional<WebCore::CAAudioStreamDescription> m_description;
        std::unique_ptr<WebCore::WebAudioBufferList> m_buffer;
        std::unique_ptr<ConsumerSharedCARingBuffer> m_ringBuffer;
        int64_t m_readOffset { 0 };
        MediaTime m_startTime;
        size_t m_frameChunkSize { 0 };

        IPC::Semaphore m_semaphore;
        RefPtr<Thread> m_thread;
        std::atomic<bool> m_shouldStopThread { false };
    };

    CheckedRef<UserMediaCaptureManager> m_manager;
    bool m_isRegisteredToParentProcessConnection { false };
    Ref<WorkQueue> m_queue;
    RefPtr<IPC::Connection> m_connection;
    // background thread member
    HashMap<WebCore::RealtimeMediaSourceIdentifier, std::unique_ptr<RemoteAudio>> m_audioSources;
    HashMap<WebCore::RealtimeMediaSourceIdentifier, Ref<RemoteRealtimeVideoSource>> m_videoSources;

    Lock m_videoFrameObjectHeapProxyLock;
    RefPtr<RemoteVideoFrameObjectHeapProxy> m_videoFrameObjectHeapProxy WTF_GUARDED_BY_LOCK(m_videoFrameObjectHeapProxyLock);
};

} // namespace WebKit

#endif
