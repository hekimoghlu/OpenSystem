/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 11, 2024.
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

#include "SourceBufferPrivate.h"

namespace WebCore {

class AudioTrackPrivate;
class InbandTextTrackPrivate;
class MockInitializationBox;
class MockMediaSourcePrivate;
class MockSampleBox;
class TimeRanges;
class VideoTrackPrivate;

class MockSourceBufferPrivate final : public SourceBufferPrivate {
public:
    static Ref<MockSourceBufferPrivate> create(MockMediaSourcePrivate&);
    virtual ~MockSourceBufferPrivate();

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::Mock; }
private:
    explicit MockSourceBufferPrivate(MockMediaSourcePrivate&);
    RefPtr<MockMediaSourcePrivate> mediaSourcePrivate() const;

    // SourceBufferPrivate overrides
    Ref<MediaPromise> appendInternal(Ref<SharedBuffer>&&) final;
    void resetParserStateInternal() final;
    bool canSetMinimumUpcomingPresentationTime(TrackID) const final;
    void setMinimumUpcomingPresentationTime(TrackID, const MediaTime&) final;
    void clearMinimumUpcomingPresentationTime(TrackID) final;
    bool canSwitchToType(const ContentType&) final;

    void flush(TrackID) final { m_enqueuedSamples.clear(); m_minimumUpcomingPresentationTime = MediaTime::invalidTime(); }
    void enqueueSample(Ref<MediaSample>&&, TrackID) final;
    bool isReadyForMoreSamples(TrackID) final { return !m_maxQueueDepth || m_enqueuedSamples.size() < m_maxQueueDepth.value(); }

    Ref<SamplesPromise> enqueuedSamplesForTrackID(TrackID) final;
    MediaTime minimumUpcomingPresentationTimeForTrackID(TrackID) final;
    void setMaximumQueueDepthForTrackID(TrackID, uint64_t) final;

    void didReceiveInitializationSegment(const MockInitializationBox&);
    void didReceiveSample(const MockSampleBox&);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    ASCIILiteral logClassName() const override { return "MockSourceBufferPrivate"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;

    const Logger& sourceBufferLogger() const final { return m_logger.get(); }
    uint64_t sourceBufferLogIdentifier() final { return logIdentifier(); }
#endif

    MediaTime m_minimumUpcomingPresentationTime;
    Vector<String> m_enqueuedSamples;
    std::optional<uint64_t> m_maxQueueDepth;
    Vector<uint8_t> m_inputBuffer;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE) && USE(AVFOUNDATION)
