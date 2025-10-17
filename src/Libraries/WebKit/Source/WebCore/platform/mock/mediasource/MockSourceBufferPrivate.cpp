/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#include "config.h"
#include "MockSourceBufferPrivate.h"

#if ENABLE(MEDIA_SOURCE)

#include "Logging.h"
#include "MediaDescription.h"
#include "MediaPlayer.h"
#include "MediaSample.h"
#include "MockBox.h"
#include "MockMediaPlayerMediaSource.h"
#include "MockMediaSourcePrivate.h"
#include "MockTracks.h"
#include "SourceBufferPrivateClient.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <wtf/NativePromise.h>
#include <wtf/StringPrintStream.h>

namespace WebCore {

class MockMediaSample final : public MediaSample {
public:
    static Ref<MockMediaSample> create(const MockSampleBox& box) { return adoptRef(*new MockMediaSample(box)); }
    virtual ~MockMediaSample() = default;

private:
    MockMediaSample(const MockSampleBox& box)
        : m_box(box)
        , m_id(box.trackID())
    {
    }

    MediaTime presentationTime() const override { return m_box.presentationTimestamp(); }
    MediaTime decodeTime() const override { return m_box.decodeTimestamp(); }
    MediaTime duration() const override { return m_box.duration(); }
    TrackID trackID() const override { return m_id; }
    size_t sizeInBytes() const override { return sizeof(m_box); }
    SampleFlags flags() const override;
    PlatformSample platformSample() const override;
    PlatformSample::Type platformSampleType() const override { return PlatformSample::MockSampleBoxType; }
    FloatSize presentationSize() const override { return FloatSize(); }
    void dump(PrintStream&) const override;
    void offsetTimestampsBy(const MediaTime& offset) override { m_box.offsetTimestampsBy(offset); }
    void setTimestamps(const MediaTime& presentationTimestamp, const MediaTime& decodeTimestamp) override { m_box.setTimestamps(presentationTimestamp, decodeTimestamp); }
    Ref<MediaSample> createNonDisplayingCopy() const override;

    unsigned generation() const { return m_box.generation(); }

    MockSampleBox m_box;
    TrackID m_id;
};

MediaSample::SampleFlags MockMediaSample::flags() const
{
    unsigned flags = None;
    if (m_box.isSync())
        flags |= IsSync;
    if (m_box.isNonDisplaying())
        flags |= IsNonDisplaying;
    return SampleFlags(flags);
}

PlatformSample MockMediaSample::platformSample() const
{
    PlatformSample sample = { PlatformSample::MockSampleBoxType, { &m_box } };
    return sample;
}

void MockMediaSample::dump(PrintStream& out) const
{
    out.print("{PTS(", presentationTime(), "), DTS(", decodeTime(), "), duration(", duration(), "), flags(", (int)flags(), "), generation(", generation(), ")}");
}

Ref<MediaSample> MockMediaSample::createNonDisplayingCopy() const
{
    auto copy = MockMediaSample::create(m_box);
    copy->m_box.setFlag(MockSampleBox::IsNonDisplaying);
    return copy;
}

class MockMediaDescription final : public MediaDescription {
public:
    static Ref<MockMediaDescription> create(const MockTrackBox& box) { return adoptRef(*new MockMediaDescription(box)); }
    virtual ~MockMediaDescription() = default;

    bool isVideo() const final { return m_box.kind() == MockTrackBox::Video; }
    bool isAudio() const final { return m_box.kind() == MockTrackBox::Audio; }
    bool isText() const final { return m_box.kind() == MockTrackBox::Text; }

private:
    MockMediaDescription(const MockTrackBox& box)
        : MediaDescription(box.codec().isolatedCopy())
        , m_box(box)
    {
    }
    const MockTrackBox m_box;
};

Ref<MockSourceBufferPrivate> MockSourceBufferPrivate::create(MockMediaSourcePrivate& parent)
{
    return adoptRef(*new MockSourceBufferPrivate(parent));
}

MockSourceBufferPrivate::MockSourceBufferPrivate(MockMediaSourcePrivate& parent)
    : SourceBufferPrivate(parent)
#if !RELEASE_LOG_DISABLED
    , m_logger(parent.logger())
    , m_logIdentifier(parent.nextSourceBufferLogIdentifier())
#endif
{
}

MockSourceBufferPrivate::~MockSourceBufferPrivate() = default;

RefPtr<MockMediaSourcePrivate> MockSourceBufferPrivate::mediaSourcePrivate() const
{
    return dynamicDowncast<MockMediaSourcePrivate>(m_mediaSource.get());
}

Ref<MediaPromise> MockSourceBufferPrivate::appendInternal(Ref<SharedBuffer>&& data)
{
    m_inputBuffer.appendVector(data->extractData());

    while (m_inputBuffer.size()) {
        auto buffer = ArrayBuffer::create(m_inputBuffer);
        uint64_t boxLength = MockBox::peekLength(buffer.ptr());
        if (boxLength > buffer->byteLength())
            break;

        String type = MockBox::peekType(buffer.ptr());
        if (type == MockInitializationBox::type()) {
            MockInitializationBox initBox = MockInitializationBox(buffer.ptr());
            didReceiveInitializationSegment(initBox);
        } else if (type == MockSampleBox::type()) {
            MockSampleBox sampleBox = MockSampleBox(buffer.ptr());
            didReceiveSample(sampleBox);
        } else {
            m_inputBuffer.clear();
            return MediaPromise::createAndReject(PlatformMediaError::ParsingError);
        }
        m_inputBuffer.remove(0, boxLength);
    }

    return MediaPromise::createAndResolve();
}

void MockSourceBufferPrivate::didReceiveInitializationSegment(const MockInitializationBox& initBox)
{
    SourceBufferPrivateClient::InitializationSegment segment;
    segment.duration = initBox.duration();

    for (auto& trackBox : initBox.tracks()) {
        if (trackBox.kind() == MockTrackBox::Video) {
            SourceBufferPrivateClient::InitializationSegment::VideoTrackInformation info;
            info.track = MockVideoTrackPrivate::create(trackBox);
            info.description = MockMediaDescription::create(trackBox);
            segment.videoTracks.append(info);
        } else if (trackBox.kind() == MockTrackBox::Audio) {
            SourceBufferPrivateClient::InitializationSegment::AudioTrackInformation info;
            info.track = MockAudioTrackPrivate::create(trackBox);
            info.description = MockMediaDescription::create(trackBox);
            segment.audioTracks.append(info);
        } else if (trackBox.kind() == MockTrackBox::Text) {
            SourceBufferPrivateClient::InitializationSegment::TextTrackInformation info;
            info.track = MockTextTrackPrivate::create(trackBox);
            info.description = MockMediaDescription::create(trackBox);
            segment.textTracks.append(info);
        }
    }

    SourceBufferPrivate::didReceiveInitializationSegment(WTFMove(segment));
}

void MockSourceBufferPrivate::didReceiveSample(const MockSampleBox& sampleBox)
{
    SourceBufferPrivate::didReceiveSample(MockMediaSample::create(sampleBox));
}

void MockSourceBufferPrivate::resetParserStateInternal()
{
}

Ref<SourceBufferPrivate::SamplesPromise> MockSourceBufferPrivate::enqueuedSamplesForTrackID(TrackID)
{
    return SamplesPromise::createAndResolve(copyToVector(m_enqueuedSamples));
}

MediaTime MockSourceBufferPrivate::minimumUpcomingPresentationTimeForTrackID(TrackID)
{
    return m_minimumUpcomingPresentationTime;
}

void MockSourceBufferPrivate::setMaximumQueueDepthForTrackID(TrackID, uint64_t maxQueueDepth)
{
    m_maxQueueDepth = maxQueueDepth;
}

bool MockSourceBufferPrivate::canSetMinimumUpcomingPresentationTime(TrackID) const
{
    return true;
}

void MockSourceBufferPrivate::setMinimumUpcomingPresentationTime(TrackID, const MediaTime& presentationTime)
{
    m_minimumUpcomingPresentationTime = presentationTime;
}

void MockSourceBufferPrivate::clearMinimumUpcomingPresentationTime(TrackID)
{
    m_minimumUpcomingPresentationTime = MediaTime::invalidTime();
}

bool MockSourceBufferPrivate::canSwitchToType(const ContentType& contentType)
{
    MediaEngineSupportParameters parameters;
    parameters.isMediaSource = true;
    parameters.type = contentType;
    return MockMediaPlayerMediaSource::supportsType(parameters) != MediaPlayer::SupportsType::IsNotSupported;
}

void MockSourceBufferPrivate::enqueueSample(Ref<MediaSample>&& sample, TrackID)
{
    RefPtr mediaSource = mediaSourcePrivate();
    if (!mediaSource)
        return;

    PlatformSample platformSample = sample->platformSample();
    if (platformSample.type != PlatformSample::MockSampleBoxType)
        return;

    auto* box = platformSample.sample.mockSampleBox;
    if (!box)
        return;

    mediaSource->incrementTotalVideoFrames();
    if (box->isCorrupted())
        mediaSource->incrementCorruptedFrames();
    if (box->isDropped())
        mediaSource->incrementDroppedFrames();
    if (box->isDelayed())
        mediaSource->incrementTotalFrameDelayBy(MediaTime(1, 1));

    m_enqueuedSamples.append(toString(sample.get()));
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& MockSourceBufferPrivate::logChannel() const
{
    return LogMediaSource;
}
#endif

}

#endif

