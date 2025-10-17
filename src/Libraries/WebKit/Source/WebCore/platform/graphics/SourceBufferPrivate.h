/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#if ENABLE(MEDIA_SOURCE)

#include "InbandTextTrackPrivate.h"
#include "MediaDescription.h"
#include "MediaPlayer.h"
#include "MediaSample.h"
#include "PlatformTimeRanges.h"
#include "SampleMap.h"
#include "SourceBufferPrivateClient.h"
#include "TimeRanges.h"
#include <optional>
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/Logger.h>
#include <wtf/LoggerHelper.h>
#include <wtf/NativePromise.h>
#include <wtf/Ref.h>
#include <wtf/StdUnorderedMap.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class MediaSourcePrivate;
class SharedBuffer;
class TrackBuffer;
class TimeRanges;

#if ENABLE(ENCRYPTED_MEDIA)
class CDMInstance;
#endif
#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
class LegacyCDMSession;
#endif

enum class SourceBufferAppendMode : uint8_t {
    Segments,
    Sequence
};

class SourceBufferPrivate
    : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<SourceBufferPrivate, WTF::DestructionThread::Main>
#if !RELEASE_LOG_DISABLED
    , public LoggerHelper
#endif
{
public:
    WEBCORE_EXPORT explicit SourceBufferPrivate(MediaSourcePrivate&);
    WEBCORE_EXPORT virtual ~SourceBufferPrivate();

    virtual constexpr MediaPlatformType platformType() const = 0;

    WEBCORE_EXPORT virtual void setActive(bool);

    WEBCORE_EXPORT virtual Ref<MediaPromise> append(Ref<SharedBuffer>&&);

    virtual void abort();
    // Overrides must call the base class.
    virtual void resetParserState();
    virtual void removedFromMediaSource();

    virtual bool canSwitchToType(const ContentType&) { return false; }

    WEBCORE_EXPORT virtual void setMediaSourceEnded(bool);
    virtual void setMode(SourceBufferAppendMode mode) { m_appendMode = mode; }
    WEBCORE_EXPORT virtual void reenqueueMediaIfNeeded(const MediaTime& currentMediaTime);
    WEBCORE_EXPORT virtual void addTrackBuffer(TrackID, RefPtr<MediaDescription>&&);
    WEBCORE_EXPORT virtual void resetTrackBuffers();
    WEBCORE_EXPORT virtual void clearTrackBuffers(bool shouldReportToClient = false);
    WEBCORE_EXPORT virtual void setAllTrackBuffersNeedRandomAccess();
    virtual void setGroupStartTimestamp(const MediaTime& mediaTime) { m_groupStartTimestamp = mediaTime; }
    virtual void setGroupStartTimestampToEndTimestamp() { m_groupStartTimestamp = m_groupEndTimestamp; }
    virtual void setShouldGenerateTimestamps(bool flag) { m_shouldGenerateTimestamps = flag; }
    WEBCORE_EXPORT virtual Ref<MediaPromise> removeCodedFrames(const MediaTime& start, const MediaTime& end, const MediaTime& currentMediaTime);
    WEBCORE_EXPORT virtual bool evictCodedFrames(uint64_t newDataSize, const MediaTime& currentTime);
    WEBCORE_EXPORT virtual void asyncEvictCodedFrames(uint64_t newDataSize, const MediaTime& currentTime);
    WEBCORE_EXPORT virtual size_t platformEvictionThreshold() const;
    WEBCORE_EXPORT virtual uint64_t totalTrackBufferSizeInBytes() const;
    WEBCORE_EXPORT virtual void resetTimestampOffsetInTrackBuffers();
    virtual void startChangingType() { m_pendingInitializationSegmentForChangeType = true; }
    virtual void setTimestampOffset(const MediaTime& timestampOffset) { m_timestampOffset = timestampOffset; }
    virtual void setAppendWindowStart(const MediaTime& appendWindowStart) { m_appendWindowStart = appendWindowStart;}
    virtual void setAppendWindowEnd(const MediaTime& appendWindowEnd) { m_appendWindowEnd = appendWindowEnd; }

    using ComputeSeekPromise = MediaTimePromise;
    WEBCORE_EXPORT virtual Ref<ComputeSeekPromise> computeSeekTime(const SeekTarget&);
    WEBCORE_EXPORT virtual void seekToTime(const MediaTime&);
    WEBCORE_EXPORT virtual void updateTrackIds(Vector<std::pair<TrackID, TrackID>>&& trackIdPairs);

    WEBCORE_EXPORT void setClient(SourceBufferPrivateClient&);

    void setMediaSourceDuration(const MediaTime& duration) { m_mediaSourceDuration = duration; }

    WEBCORE_EXPORT virtual bool isBufferFullFor(uint64_t requiredSize) const;
    WEBCORE_EXPORT virtual bool canAppend(uint64_t requiredSize) const;
    SourceBufferEvictionData evictionData() const { return m_evictionData; }
    WEBCORE_EXPORT Vector<PlatformTimeRanges> trackBuffersRanges() const;

    // Methods used by MediaSourcePrivate
    bool hasAudio() const { return m_hasAudio; }
    bool hasVideo() const { return m_hasVideo; }
    bool hasReceivedFirstInitializationSegment() const { return m_receivedFirstInitializationSegment; }

    virtual MediaTime timestampOffset() const { return m_timestampOffset; }

    virtual size_t platformMaximumBufferSize() const { return 0; }
    virtual Ref<GenericPromise> setMaximumBufferSize(size_t);

    // Methods for ManagedSourceBuffer
    WEBCORE_EXPORT virtual void memoryPressure(const MediaTime& currentTime);

    // Methods for Detachable MediaSource
    virtual void detach() { }
    WEBCORE_EXPORT virtual void attach();

    // Internals Utility methods
    using SamplesPromise = NativePromise<Vector<String>, PlatformMediaError>;
    WEBCORE_EXPORT virtual Ref<SamplesPromise> bufferedSamplesForTrackId(TrackID);
    WEBCORE_EXPORT virtual Ref<SamplesPromise> enqueuedSamplesForTrackID(TrackID);
    virtual MediaTime minimumUpcomingPresentationTimeForTrackID(TrackID) { return MediaTime::invalidTime(); }
    virtual void setMaximumQueueDepthForTrackID(TrackID, uint64_t) { }

#if !RELEASE_LOG_DISABLED
    virtual const Logger& sourceBufferLogger() const = 0;
    virtual uint64_t sourceBufferLogIdentifier() = 0;
#endif

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
    virtual void setCDMSession(LegacyCDMSession*) { }
#endif
#if ENABLE(ENCRYPTED_MEDIA)
    virtual void setCDMInstance(CDMInstance*) { }
    virtual bool waitingForKey() const { return false; }
    virtual void attemptToDecrypt() { }
#endif

protected:
    WEBCORE_EXPORT explicit SourceBufferPrivate(MediaSourcePrivate&, GuaranteedSerialFunctionDispatcher&);
    MediaTime currentTime() const;
    MediaTime mediaSourceDuration() const;

    WEBCORE_EXPORT void ensureOnDispatcher(Function<void()>&&) const;

    using InitializationSegment = SourceBufferPrivateClient::InitializationSegment;
    WEBCORE_EXPORT void didReceiveInitializationSegment(InitializationSegment&&);
    WEBCORE_EXPORT void didUpdateFormatDescriptionForTrackId(Ref<TrackInfo>&&, uint64_t);
    WEBCORE_EXPORT void didReceiveSample(Ref<MediaSample>&&);

    virtual Ref<MediaPromise> appendInternal(Ref<SharedBuffer>&&) = 0;
    virtual void resetParserStateInternal() = 0;
    virtual MediaTime timeFudgeFactor() const { return PlatformTimeRanges::timeFudgeFactor(); }
    virtual bool isActive() const { return m_isActive; }
    virtual bool isSeeking() const { return false; }
    virtual void flush(TrackID) { }
    virtual void enqueueSample(Ref<MediaSample>&&, TrackID) { }
    virtual void allSamplesInTrackEnqueued(TrackID) { }
    virtual bool isReadyForMoreSamples(TrackID) { return false; }
    virtual void notifyClientWhenReadyForMoreSamples(TrackID) { }

    virtual bool canSetMinimumUpcomingPresentationTime(TrackID) const { return false; }
    virtual void setMinimumUpcomingPresentationTime(TrackID, const MediaTime&) { }
    virtual void clearMinimumUpcomingPresentationTime(TrackID) { }

    enum class NeedsFlush: bool {
        No = 0,
        Yes
    };

    void reenqueSamples(TrackID, NeedsFlush = NeedsFlush::Yes);

    virtual bool precheckInitializationSegment(const InitializationSegment&) { return true; }
    virtual void processInitializationSegment(std::optional<InitializationSegment>&&) { }
    virtual void processFormatDescriptionForTrackId(Ref<TrackInfo>&&, uint64_t) { }

    void provideMediaData(TrackID);

    virtual bool isMediaSampleAllowed(const MediaSample&) const { return true; }

    // Must be called once all samples have been processed.
    WEBCORE_EXPORT void appendCompleted(bool parsingSucceeded, Function<void()>&& = [] { });

    WEBCORE_EXPORT RefPtr<SourceBufferPrivateClient> client() const;

    ThreadSafeWeakPtr<MediaSourcePrivate> m_mediaSource { nullptr };
    const Ref<GuaranteedSerialFunctionDispatcher> m_dispatcher; // SerialFunctionDispatcher the SourceBufferPrivate/MediaSourcePrivate

    SourceBufferEvictionData m_evictionData;

private:
    MediaTime minimumBufferedTime() const;
    MediaTime maximumBufferedTime() const;
    Ref<MediaPromise> updateBuffered();
    void updateHighestPresentationTimestamp();
    void updateMinimumUpcomingPresentationTime(TrackBuffer&, TrackID);
    void reenqueueMediaForTime(TrackBuffer&, TrackID, const MediaTime&, NeedsFlush = NeedsFlush::Yes);
    bool validateInitializationSegment(const InitializationSegment&);
    void provideMediaData(TrackBuffer&, TrackID);
    void setBufferedDirty(bool);
    void trySignalAllSamplesInTrackEnqueued(TrackBuffer&, TrackID);
    MediaTime findPreviousSyncSamplePresentationTime(const MediaTime&);
    void removeCodedFramesInternal(const MediaTime& start, const MediaTime& end, const MediaTime& currentMediaTime);
    bool evictFrames(uint64_t newDataSize, const MediaTime& currentTime);
    bool hasTooManySamples() const;
    void iterateTrackBuffers(Function<void(TrackBuffer&)>&&);
    void iterateTrackBuffers(Function<void(const TrackBuffer&)>&&) const;

    bool m_hasAudio { false };
    bool m_hasVideo { false };
    bool m_isActive { false };

    ThreadSafeWeakPtr<SourceBufferPrivateClient> m_client;

    StdUnorderedMap<TrackID, UniqueRef<TrackBuffer>> m_trackBufferMap;
    SourceBufferAppendMode m_appendMode { SourceBufferAppendMode::Segments };

    using OperationPromise = NativePromise<void, PlatformMediaError, WTF::PromiseOption::Default | WTF::PromiseOption::NonExclusive>;
    Ref<OperationPromise> m_currentSourceBufferOperation { OperationPromise::createAndResolve() };

    bool m_shouldGenerateTimestamps { false };
    bool m_receivedFirstInitializationSegment { false };
    bool m_pendingInitializationSegmentForChangeType { false };
    size_t m_abortCount { 0 };

    void processPendingMediaSamples();
    bool processMediaSample(SourceBufferPrivateClient&, Ref<MediaSample>&&);

    enum class ComputeEvictionDataRule {
        Default,
        ForceNotification
    };
    void computeEvictionData(ComputeEvictionDataRule = ComputeEvictionDataRule::Default);

    using SamplesVector = Vector<Ref<MediaSample>>;
    SamplesVector m_pendingSamples;
    Ref<MediaPromise> m_currentAppendProcessing { MediaPromise::createAndResolve() };

    MediaTime m_timestampOffset;
    MediaTime m_appendWindowStart { MediaTime::zeroTime() };
    MediaTime m_appendWindowEnd { MediaTime::positiveInfiniteTime() };
    MediaTime m_highestPresentationTimestamp;
    MediaTime m_mediaSourceDuration { MediaTime::invalidTime() };

    MediaTime m_groupStartTimestamp { MediaTime::invalidTime() };
    MediaTime m_groupEndTimestamp { MediaTime::zeroTime() };

    bool m_isMediaSourceEnded { false };
    std::optional<InitializationSegment> m_lastInitializationSegment;
};

} // namespace WebCore

#endif
