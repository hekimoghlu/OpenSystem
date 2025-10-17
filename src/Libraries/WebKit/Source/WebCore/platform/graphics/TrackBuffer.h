/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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

#include "MediaDescription.h"
#include "MediaSample.h"
#include "PlatformTimeRanges.h"
#include "SampleMap.h"
#include <wtf/Logger.h>
#include <wtf/LoggerHelper.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class TrackBuffer final
#if !RELEASE_LOG_DISABLED
    : public LoggerHelper
#endif
{
    WTF_MAKE_TZONE_ALLOCATED(TrackBuffer);
public:
    static UniqueRef<TrackBuffer> create(RefPtr<MediaDescription>&&);
    static UniqueRef<TrackBuffer> create(RefPtr<MediaDescription>&&, const MediaTime&);
    
    MediaTime maximumBufferedTime() const;
    void addBufferedRange(const MediaTime& start, const MediaTime& end, AddTimeRangeOption = AddTimeRangeOption::None);
    void addSample(MediaSample&);
    
    bool updateMinimumUpcomingPresentationTime();
    
    bool reenqueueMediaForTime(const MediaTime&, const MediaTime& timeFudgeFactor, bool isEnded = false);
    MediaTime findSeekTimeForTargetTime(const MediaTime& targetTime, const MediaTime& negativeThreshold, const MediaTime& positiveThreshold);
    int64_t removeCodedFrames(const MediaTime& start, const MediaTime& end, const MediaTime& currentTime);
    PlatformTimeRanges removeSamples(const DecodeOrderSampleMap::MapType&, ASCIILiteral);
    int64_t codedFramesIntervalSize(const MediaTime& start, const MediaTime& end);

    void resetTimestampOffset();
    void reset();
    void clearSamples();
    
    const MediaTime& lastDecodeTimestamp() const { return m_lastDecodeTimestamp; }
    void setLastDecodeTimestamp(MediaTime timestamp) { m_lastDecodeTimestamp = WTFMove(timestamp); }
    
    const MediaTime& greatestFrameDuration() const { return m_greatestFrameDuration; }
    void setGreatestFrameDuration(MediaTime duration) { m_greatestFrameDuration = WTFMove(duration); }
    const MediaTime& lastFrameDuration() const { return m_lastFrameDuration; }
    void setLastFrameDuration(MediaTime duration) { m_lastFrameDuration = WTFMove(duration); }
    
    const MediaTime& highestPresentationTimestamp() const { return m_highestPresentationTimestamp; }
    void setHighestPresentationTimestamp(MediaTime timestamp) { m_highestPresentationTimestamp = WTFMove(timestamp); }
    
    const MediaTime& highestEnqueuedPresentationTime() const { return m_highestEnqueuedPresentationTime; }
    void setHighestEnqueuedPresentationTime(MediaTime timestamp) { m_highestEnqueuedPresentationTime = WTFMove(timestamp); }
    const MediaTime& minimumEnqueuedPresentationTime() const { return m_minimumEnqueuedPresentationTime; }
    void setMinimumEnqueuedPresentationTime(MediaTime timestamp) { m_minimumEnqueuedPresentationTime = WTFMove(timestamp); }
    
    const DecodeOrderSampleMap::KeyType& lastEnqueuedDecodeKey() const { return m_lastEnqueuedDecodeKey; }
    void setLastEnqueuedDecodeKey(DecodeOrderSampleMap::KeyType key) { m_lastEnqueuedDecodeKey = WTFMove(key); }
    
    const MediaTime& enqueueDiscontinuityBoundary() const { return m_enqueueDiscontinuityBoundary; }
    void setEnqueueDiscontinuityBoundary(MediaTime boundary) { m_enqueueDiscontinuityBoundary = WTFMove(boundary); }
    
    const MediaTime& roundedTimestampOffset() const { return m_roundedTimestampOffset; }
    void setRoundedTimestampOffset(MediaTime offset) { m_roundedTimestampOffset = WTFMove(offset); }
    void setRoundedTimestampOffset(const MediaTime&, uint32_t, const MediaTime&);
    
    uint32_t lastFrameTimescale() const { return m_lastFrameTimescale; }
    void setLastFrameTimescale(uint32_t timescale) { m_lastFrameTimescale = timescale; }
    bool needRandomAccessFlag() const { return m_needRandomAccessFlag; }
    void setNeedRandomAccessFlag(bool flag) { m_needRandomAccessFlag = flag; }
    bool enabled() const { return m_enabled; }
    void setEnabled(bool enabled) { m_enabled = enabled; }
    bool needsReenqueueing() const { return m_needsReenqueueing; }
    void setNeedsReenqueueing(bool flag) { m_needsReenqueueing = flag; }
    bool needsMinimumUpcomingPresentationTimeUpdating() const { return m_needsMinimumUpcomingPresentationTimeUpdating; }
    void setNeedsMinimumUpcomingPresentationTimeUpdating(bool flag) { m_needsMinimumUpcomingPresentationTimeUpdating = flag; }
    
    const SampleMap& samples() const { return m_samples; }
    SampleMap& samples() { return m_samples; }
    const DecodeOrderSampleMap::MapType& decodeQueue() const { return m_decodeQueue; }
    DecodeOrderSampleMap::MapType& decodeQueue() { return m_decodeQueue; }
    const RefPtr<MediaDescription>& description() const { return m_description; }
    const PlatformTimeRanges& buffered() const { return m_buffered; }
    PlatformTimeRanges& buffered() { return m_buffered; }
    
#if !RELEASE_LOG_DISABLED
    void setLogger(const Logger&, uint64_t);
    const Logger& logger() const final { ASSERT(m_logger); return *m_logger.get(); }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    ASCIILiteral logClassName() const final { return "TrackBuffer"_s; }
    WTFLogChannel& logChannel() const final;
#endif
    
private:
    friend UniqueRef<TrackBuffer> WTF::makeUniqueRefWithoutFastMallocCheck<TrackBuffer>(RefPtr<WebCore::MediaDescription>&&, const WTF::MediaTime&);
    TrackBuffer(RefPtr<MediaDescription>&&, const MediaTime&);
    
    SampleMap m_samples;
    DecodeOrderSampleMap::MapType m_decodeQueue;
    RefPtr<MediaDescription> m_description;
    PlatformTimeRanges m_buffered;
    
    MediaTime m_lastDecodeTimestamp { MediaTime::invalidTime() };
    
    MediaTime m_greatestFrameDuration { MediaTime::invalidTime() };
    MediaTime m_lastFrameDuration { MediaTime::invalidTime() };
    
    MediaTime m_highestPresentationTimestamp { MediaTime::invalidTime() };
    
    MediaTime m_highestEnqueuedPresentationTime { MediaTime::invalidTime() };
    MediaTime m_minimumEnqueuedPresentationTime { MediaTime::invalidTime() };
    
    DecodeOrderSampleMap::KeyType m_lastEnqueuedDecodeKey { MediaTime::invalidTime(), MediaTime::invalidTime() };
    
    MediaTime m_enqueueDiscontinuityBoundary;
    MediaTime m_discontinuityTolerance;
    
    MediaTime m_roundedTimestampOffset { MediaTime::invalidTime() };
    
#if !RELEASE_LOG_DISABLED
    RefPtr<const Logger> m_logger;
    uint64_t m_logIdentifier { 0 };
#endif
    
    uint32_t m_lastFrameTimescale { 0 };
    bool m_needRandomAccessFlag { true };
    bool m_enabled { false };
    bool m_needsReenqueueing { false };
    bool m_needsMinimumUpcomingPresentationTimeUpdating { false };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE)
