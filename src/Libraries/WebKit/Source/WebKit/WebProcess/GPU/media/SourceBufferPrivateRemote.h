/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

#if ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)

#include "GPUProcessConnection.h"
#include "RemoteSourceBufferIdentifier.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/ContentType.h>
#include <WebCore/MediaSample.h>
#include <WebCore/SourceBufferPrivate.h>
#include <WebCore/SourceBufferPrivateClient.h>
#include <atomic>
#include <wtf/Lock.h>
#include <wtf/LoggerHelper.h>
#include <wtf/MediaTime.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
}

namespace WebCore {
class PlatformTimeRanges;
}

namespace WebKit {

struct InitializationSegmentInfo;
class MediaPlayerPrivateRemote;
class MediaSourcePrivateRemote;

class SourceBufferPrivateRemote final
    : public WebCore::SourceBufferPrivate
{
    WTF_MAKE_TZONE_ALLOCATED(SourceBufferPrivateRemote);
public:
    static Ref<SourceBufferPrivateRemote> create(GPUProcessConnection&, RemoteSourceBufferIdentifier, MediaSourcePrivateRemote&);
    virtual ~SourceBufferPrivateRemote();

    constexpr WebCore::MediaPlatformType platformType() const final { return WebCore::MediaPlatformType::Remote; }

    static WorkQueue& queue();

    class MessageReceiver : public IPC::WorkQueueMessageReceiver {
    public:
        static Ref<MessageReceiver> create(SourceBufferPrivateRemote& parent)
        {
            return adoptRef(*new MessageReceiver(parent));
        }

    private:
        MessageReceiver(SourceBufferPrivateRemote&);
        void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
        void sourceBufferPrivateDidReceiveInitializationSegment(InitializationSegmentInfo&&, CompletionHandler<void(WebCore::MediaPromise::Result&&)>&&);
        void takeOwnershipOfMemory(WebCore::SharedMemory::Handle&&);
        void sourceBufferPrivateHighestPresentationTimestampChanged(const MediaTime&);
        void sourceBufferPrivateBufferedChanged(Vector<WebCore::PlatformTimeRanges>&&, CompletionHandler<void()>&&);
        void sourceBufferPrivateDurationChanged(const MediaTime&, CompletionHandler<void()>&&);
        void sourceBufferPrivateDidDropSample();
        void sourceBufferPrivateDidReceiveRenderingError(int64_t errorCode);
        void sourceBufferPrivateEvictionDataChanged(WebCore::SourceBufferEvictionData&&);
        void sourceBufferPrivateDidAttach(InitializationSegmentInfo&&, CompletionHandler<void(WebCore::MediaPromise::Result&&)>&&);
        RefPtr<WebCore::SourceBufferPrivateClient> client() const;
        WebCore::SourceBufferPrivateClient::InitializationSegment createInitializationSegment(MediaPlayerPrivateRemote&, InitializationSegmentInfo&&) const;
        ThreadSafeWeakPtr<SourceBufferPrivateRemote> m_parent;
    };

private:
    SourceBufferPrivateRemote(GPUProcessConnection&, RemoteSourceBufferIdentifier, MediaSourcePrivateRemote&);

    // SourceBufferPrivate overrides
    void setActive(bool) final;
    bool isActive() const final;
    Ref<WebCore::MediaPromise> append(Ref<WebCore::SharedBuffer>&&) final;
    Ref<WebCore::MediaPromise> appendInternal(Ref<WebCore::SharedBuffer>&&) final;
    void resetParserStateInternal() final;
    void abort() final;
    void resetParserState() final;
    void removedFromMediaSource() final;
    bool canSwitchToType(const WebCore::ContentType&) final;
    void setMediaSourceEnded(bool) final;
    void setMode(WebCore::SourceBufferAppendMode) final;
    void reenqueueMediaIfNeeded(const MediaTime& currentMediaTime) final;
    void addTrackBuffer(TrackID, RefPtr<WebCore::MediaDescription>&&) final;
    void resetTrackBuffers() final;
    void clearTrackBuffers(bool) final;
    void setAllTrackBuffersNeedRandomAccess() final;
    void setGroupStartTimestamp(const MediaTime&) final;
    void setGroupStartTimestampToEndTimestamp() final;
    void setShouldGenerateTimestamps(bool) final;
    Ref<WebCore::MediaPromise> removeCodedFrames(const MediaTime& start, const MediaTime& end, const MediaTime& currentMediaTime) final;
    bool evictCodedFrames(uint64_t newDataSize, const MediaTime& currentTime) final;
    void resetTimestampOffsetInTrackBuffers() final;
    void startChangingType() final;
    void setTimestampOffset(const MediaTime&) final;
    MediaTime timestampOffset() const final;
    void setAppendWindowStart(const MediaTime&) final;
    void setAppendWindowEnd(const MediaTime&) final;
    Ref<GenericPromise> setMaximumBufferSize(size_t) final;
    bool isBufferFullFor(uint64_t requiredSize) const final;
    bool canAppend(uint64_t requiredSize) const final;

    Ref<ComputeSeekPromise> computeSeekTime(const WebCore::SeekTarget&) final;
    void seekToTime(const MediaTime&) final;

    void updateTrackIds(Vector<std::pair<TrackID, TrackID>>&&) final;
    uint64_t totalTrackBufferSizeInBytes() const final;

    void memoryPressure(const MediaTime& currentTime) final;

    void detach() final;
    void attach() final;

    // Internals Utility methods
    Ref<SamplesPromise> bufferedSamplesForTrackId(TrackID) final;
    Ref<SamplesPromise> enqueuedSamplesForTrackID(TrackID) final;
    MediaTime minimumUpcomingPresentationTimeForTrackID(TrackID) final;
    void setMaximumQueueDepthForTrackID(TrackID, uint64_t) final;

    void ensureOnDispatcherSync(Function<void()>&&);
    void ensureWeakOnDispatcher(Function<void()>&&);

    RefPtr<MediaPlayerPrivateRemote> player() const;

    template<typename PC = IPC::Connection::NoOpPromiseConverter, typename T>
    auto sendWithPromisedReply(T&& message)
    {
        return m_gpuProcessConnection.get()->connection().sendWithPromisedReply<PC, T>(std::forward<T>(message), m_remoteSourceBufferIdentifier);
    }

    friend class MessageReceiver;
    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    Ref<MessageReceiver> m_receiver;
    const RemoteSourceBufferIdentifier m_remoteSourceBufferIdentifier;

    std::atomic<uint64_t> m_totalTrackBufferSizeInBytes = { 0 };

    bool isGPURunning() const { return !m_removed; }
    std::atomic<bool> m_removed { false };

    mutable Lock m_lock;
    // We mirror some members from the base class, as we require them to be atomic.
    MediaTime m_timestampOffset WTF_GUARDED_BY_LOCK(m_lock);
    std::atomic<bool> m_isActive { false };

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    ASCIILiteral logClassName() const override { return "SourceBufferPrivateRemote"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;
    const Logger& sourceBufferLogger() const final { return m_logger.get(); }
    uint64_t sourceBufferLogIdentifier() final { return logIdentifier(); }

    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::SourceBufferPrivateRemote)
static bool isType(const WebCore::SourceBufferPrivate& sourceBuffer) { return sourceBuffer.platformType() == WebCore::MediaPlatformType::Remote; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)
