/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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
#include "MockMediaPlayerMediaSource.h"

#if ENABLE(MEDIA_SOURCE)

#include "MediaPlayer.h"
#include "MediaSourcePrivate.h"
#include "MediaSourcePrivateClient.h"
#include "MockMediaSourcePrivate.h"
#include <wtf/MainThread.h>
#include <wtf/NativePromise.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class MediaPlayerFactoryMediaSourceMock final : public MediaPlayerFactory {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(MediaPlayerFactoryMediaSourceMock);
private:
    MediaPlayerEnums::MediaEngineIdentifier identifier() const final { return MediaPlayerEnums::MediaEngineIdentifier::MockMSE; };

    Ref<MediaPlayerPrivateInterface> createMediaEnginePlayer(MediaPlayer* player) const final { return adoptRef(*new MockMediaPlayerMediaSource(player)); }

    void getSupportedTypes(HashSet<String>& types) const final
    {
        return MockMediaPlayerMediaSource::getSupportedTypes(types);
    }

    MediaPlayer::SupportsType supportsTypeAndCodecs(const MediaEngineSupportParameters& parameters) const final
    {
        return MockMediaPlayerMediaSource::supportsType(parameters);
    }
};

// MediaPlayer Enigne Support
void MockMediaPlayerMediaSource::registerMediaEngine(MediaEngineRegistrar registrar)
{
    registrar(makeUnique<MediaPlayerFactoryMediaSourceMock>());
}

// FIXME: What does the word "cache" mean here?
static const HashSet<String>& mimeTypeCache()
{
    static NeverDestroyed cache = HashSet<String> {
        "video/mock"_s,
        "audio/mock"_s,
    };
    return cache;
}

void MockMediaPlayerMediaSource::getSupportedTypes(HashSet<String>& supportedTypes)
{
    supportedTypes = mimeTypeCache();
}

MediaPlayer::SupportsType MockMediaPlayerMediaSource::supportsType(const MediaEngineSupportParameters& parameters)
{
    if (!parameters.isMediaSource)
        return MediaPlayer::SupportsType::IsNotSupported;

    auto containerType = parameters.type.containerType().convertToASCIILowercase();
    if (containerType.isEmpty() || !mimeTypeCache().contains(containerType))
        return MediaPlayer::SupportsType::IsNotSupported;

    auto codecs = parameters.type.parameter(ContentType::codecsParameter());
    if (codecs.isEmpty())
        return MediaPlayer::SupportsType::MayBeSupported;

    if (codecs == "mock"_s || codecs == "kcom"_s)
        return MediaPlayer::SupportsType::IsSupported;

    return MediaPlayer::SupportsType::MayBeSupported;
}

MockMediaPlayerMediaSource::MockMediaPlayerMediaSource(MediaPlayer* player)
    : m_player(player)
{
}

MockMediaPlayerMediaSource::~MockMediaPlayerMediaSource() = default;

void MockMediaPlayerMediaSource::load(const String&)
{
    ASSERT_NOT_REACHED();
}

void MockMediaPlayerMediaSource::load(const URL&, const ContentType&, MediaSourcePrivateClient& source)
{
    if (RefPtr mediaSourcePrivate = downcast<MockMediaSourcePrivate>(source.mediaSourcePrivate())) {
        mediaSourcePrivate->setPlayer(this);
        m_mediaSourcePrivate = WTFMove(mediaSourcePrivate);
        source.reOpen();
    } else
        m_mediaSourcePrivate = MockMediaSourcePrivate::create(*this, source);
}

void MockMediaPlayerMediaSource::cancelLoad()
{
}

void MockMediaPlayerMediaSource::play()
{
    m_playing = 1;
    callOnMainThread([protectedThis = Ref { *this }] {
        protectedThis->advanceCurrentTime();
    });
}

void MockMediaPlayerMediaSource::pause()
{
    m_playing = 0;
}

FloatSize MockMediaPlayerMediaSource::naturalSize() const
{
    return FloatSize();
}

bool MockMediaPlayerMediaSource::hasVideo() const
{
    return m_mediaSourcePrivate ? m_mediaSourcePrivate->hasVideo() : false;
}

bool MockMediaPlayerMediaSource::hasAudio() const
{
    return m_mediaSourcePrivate ? m_mediaSourcePrivate->hasAudio() : false;
}

void MockMediaPlayerMediaSource::setPageIsVisible(bool)
{
}

bool MockMediaPlayerMediaSource::seeking() const
{
    return !!m_lastSeekTarget;
}

bool MockMediaPlayerMediaSource::paused() const
{
    return !m_playing;
}

MediaPlayer::NetworkState MockMediaPlayerMediaSource::networkState() const
{
    return m_networkState;
}

MediaPlayer::ReadyState MockMediaPlayerMediaSource::readyState() const
{
    return m_readyState;
}

MediaTime MockMediaPlayerMediaSource::maxTimeSeekable() const
{
    return m_duration;
}

const PlatformTimeRanges& MockMediaPlayerMediaSource::buffered() const
{
    ASSERT_NOT_REACHED();
    return PlatformTimeRanges::emptyRanges();
}

bool MockMediaPlayerMediaSource::didLoadingProgress() const
{
    return false;
}

void MockMediaPlayerMediaSource::setPresentationSize(const IntSize&)
{
}

void MockMediaPlayerMediaSource::paint(GraphicsContext&, const FloatRect&)
{
}

MediaTime MockMediaPlayerMediaSource::currentTime() const
{
    return m_lastSeekTarget ? m_lastSeekTarget->time : m_currentTime;
}

bool MockMediaPlayerMediaSource::timeIsProgressing() const
{
    return m_playing && m_mediaSourcePrivate && m_mediaSourcePrivate->hasFutureTime(currentTime());
}

void MockMediaPlayerMediaSource::notifyActiveSourceBuffersChanged()
{
    if (auto player = m_player.get())
        player->activeSourceBuffersChanged();
}

MediaTime MockMediaPlayerMediaSource::duration() const
{
    return m_mediaSourcePrivate ? m_mediaSourcePrivate->duration() : MediaTime::zeroTime();
}

void MockMediaPlayerMediaSource::seekToTarget(const SeekTarget& target)
{
    m_lastSeekTarget = target;
    m_mediaSourcePrivate->waitForTarget(target)->whenSettled(RunLoop::current(), [this, weakThis = WeakPtr { this }](auto&& result) {
        if (!weakThis || !result)
            return;

        m_mediaSourcePrivate->seekToTime(*result)->whenSettled(RunLoop::current(), [this, weakThis, seekTime = *result] {
            RefPtr protectedThis = weakThis.get();
            if (!protectedThis)
                return;
            m_lastSeekTarget.reset();
            m_currentTime = seekTime;

            if (auto player = m_player.get()) {
                player->seeked(seekTime);
                player->timeChanged();
            }

            if (m_playing) {
                callOnMainThread([this, protectedThis = WTFMove(protectedThis)] {
                    advanceCurrentTime();
                });
            }
        });
    });
}

void MockMediaPlayerMediaSource::advanceCurrentTime()
{
    if (!m_mediaSourcePrivate)
        return;

    auto buffered = m_mediaSourcePrivate->buffered();
    size_t pos = buffered.find(m_currentTime);
    if (pos == notFound)
        return;

    bool ignoreError;
    m_currentTime = std::min(m_duration, buffered.end(pos, ignoreError));
    if (auto player = m_player.get())
        player->timeChanged();
}

void MockMediaPlayerMediaSource::updateDuration(const MediaTime& duration)
{
    if (m_duration == duration)
        return;

    m_duration = duration;
    if (auto player = m_player.get())
        player->durationChanged();
}

void MockMediaPlayerMediaSource::setReadyState(MediaPlayer::ReadyState readyState)
{
    if (readyState == m_readyState)
        return;

    m_readyState = readyState;
    if (auto player = m_player.get())
        player->readyStateChanged();
}

void MockMediaPlayerMediaSource::setNetworkState(MediaPlayer::NetworkState networkState)
{
    if (networkState == m_networkState)
        return;

    m_networkState = networkState;
    if (auto player = m_player.get())
        player->networkStateChanged();
}

std::optional<VideoPlaybackQualityMetrics> MockMediaPlayerMediaSource::videoPlaybackQualityMetrics()
{
    return m_mediaSourcePrivate ? m_mediaSourcePrivate->videoPlaybackQualityMetrics() : std::nullopt;
}

DestinationColorSpace MockMediaPlayerMediaSource::colorSpace()
{
    return DestinationColorSpace::SRGB();
}

}

#endif
