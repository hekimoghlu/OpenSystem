/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 2, 2023.
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

#include "GPUConnectionToWebProcess.h"
#include "MessageReceiver.h"
#include "RemoteSourceBufferIdentifier.h"
#include <WebCore/MediaDescription.h>
#include <WebCore/SharedMemory.h>
#include <WebCore/SourceBufferPrivate.h>
#include <WebCore/SourceBufferPrivateClient.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

namespace IPC {
class Connection;
class Decoder;
class SharedBufferReference;
}

namespace WebCore {
class ContentType;
class MediaSample;
class PlatformTimeRanges;
}

namespace WebKit {

struct MediaDescriptionInfo;
class RemoteMediaPlayerProxy;

class RemoteSourceBufferProxy final
    : public WebCore::SourceBufferPrivateClient
    , private IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteSourceBufferProxy);
public:
    static Ref<RemoteSourceBufferProxy> create(GPUConnectionToWebProcess&, RemoteSourceBufferIdentifier, Ref<WebCore::SourceBufferPrivate>&&, RemoteMediaPlayerProxy&);
    virtual ~RemoteSourceBufferProxy();

    void ref() const final { WebCore::SourceBufferPrivateClient::ref(); }
    void deref() const final { WebCore::SourceBufferPrivateClient::deref(); }

    void setMediaPlayer(RemoteMediaPlayerProxy&);

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    RemoteSourceBufferProxy(GPUConnectionToWebProcess&, RemoteSourceBufferIdentifier, Ref<WebCore::SourceBufferPrivate>&&, RemoteMediaPlayerProxy&);

    RefPtr<IPC::Connection> connection() const;
    Ref<WebCore::SourceBufferPrivate> protectedSourceBufferPrivate() const { return m_sourceBufferPrivate; }

    // SourceBufferPrivateClient
    Ref<WebCore::MediaPromise> sourceBufferPrivateDidReceiveInitializationSegment(InitializationSegment&&) final;
    Ref<WebCore::MediaPromise> sourceBufferPrivateBufferedChanged(const Vector<WebCore::PlatformTimeRanges>&) final;
    void sourceBufferPrivateHighestPresentationTimestampChanged(const MediaTime&) final;
    Ref<WebCore::MediaPromise> sourceBufferPrivateDurationChanged(const MediaTime&) final;
    void sourceBufferPrivateDidDropSample() final;
    void sourceBufferPrivateDidReceiveRenderingError(int64_t errorCode) final;
    void sourceBufferPrivateEvictionDataChanged(const WebCore::SourceBufferEvictionData&) final;
    Ref<WebCore::MediaPromise> sourceBufferPrivateDidAttach(InitializationSegment&&) final;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;
    bool didReceiveSyncMessage(IPC::Connection&, IPC::Decoder&, UniqueRef<IPC::Encoder>&) final;

    void setActive(bool);
    void canSwitchToType(const WebCore::ContentType&, CompletionHandler<void(bool)>&&);
    void setMode(WebCore::SourceBufferAppendMode);
    void append(IPC::SharedBufferReference&&, CompletionHandler<void(WebCore::MediaPromise::Result, const MediaTime&)>&&);
    void abort();
    void resetParserState();
    void removedFromMediaSource();
    void setMediaSourceEnded(bool);
    void startChangingType();
    void removeCodedFrames(const MediaTime& start, const MediaTime& end, const MediaTime& currentTime, CompletionHandler<void()>&&);
    void evictCodedFrames(uint64_t newDataSize, const MediaTime& currentTime, CompletionHandler<void(Vector<WebCore::PlatformTimeRanges>&&, WebCore::SourceBufferEvictionData&&)>&&);
    void asyncEvictCodedFrames(uint64_t newDataSize, const MediaTime& currentTime);
    void addTrackBuffer(TrackID);
    void resetTrackBuffers();
    void clearTrackBuffers();
    void setAllTrackBuffersNeedRandomAccess();
    void reenqueueMediaIfNeeded(const MediaTime& currentMediaTime);
    void setGroupStartTimestamp(const MediaTime&);
    void setGroupStartTimestampToEndTimestamp();
    void setShouldGenerateTimestamps(bool);
    void resetTimestampOffsetInTrackBuffers();
    void setTimestampOffset(const MediaTime&);
    void setAppendWindowStart(const MediaTime&);
    void setAppendWindowEnd(const MediaTime&);
    void setMaximumBufferSize(size_t, CompletionHandler<void()>&&);
    void computeSeekTime(const WebCore::SeekTarget&, CompletionHandler<void(WebCore::SourceBufferPrivate::ComputeSeekPromise::Result&&)>&&);
    void seekToTime(const MediaTime&);
    void updateTrackIds(Vector<std::pair<TrackID, TrackID>>&&);
    void bufferedSamplesForTrackId(TrackID, CompletionHandler<void(WebCore::SourceBufferPrivate::SamplesPromise::Result&&)>&&);
    void enqueuedSamplesForTrackID(TrackID, CompletionHandler<void(WebCore::SourceBufferPrivate::SamplesPromise::Result&&)>&&);
    void memoryPressure(const MediaTime& currentTime);
    void minimumUpcomingPresentationTimeForTrackID(TrackID, CompletionHandler<void(MediaTime)>&&);
    void setMaximumQueueDepthForTrackID(TrackID, uint64_t);
    void detach();
    void attach();
    void disconnect();
    std::optional<InitializationSegmentInfo> createInitializationSegmentInfo(InitializationSegment&&);

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    RemoteSourceBufferIdentifier m_identifier;
    Ref<WebCore::SourceBufferPrivate> m_sourceBufferPrivate;
    WeakPtr<RemoteMediaPlayerProxy> m_remoteMediaPlayerProxy;

    StdUnorderedMap<TrackID, Ref<WebCore::MediaDescription>> m_mediaDescriptions;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(MEDIA_SOURCE)
