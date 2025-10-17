/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 21, 2024.
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
#include "MockMediaSourcePrivate.h"

#if ENABLE(MEDIA_SOURCE)

#include "ContentType.h"
#include "Logging.h"
#include "MediaSourcePrivateClient.h"
#include "MockMediaPlayerMediaSource.h"
#include "MockSourceBufferPrivate.h"

namespace WebCore {

Ref<MockMediaSourcePrivate> MockMediaSourcePrivate::create(MockMediaPlayerMediaSource& parent, MediaSourcePrivateClient& client)
{
    auto source = adoptRef(*new MockMediaSourcePrivate(parent, client));
    client.setPrivateAndOpen(source.copyRef());
    return source;
}

MockMediaSourcePrivate::MockMediaSourcePrivate(MockMediaPlayerMediaSource& parent, MediaSourcePrivateClient& client)
    : MediaSourcePrivate(client)
    , m_player(parent)
#if !RELEASE_LOG_DISABLED
    , m_logger(m_player->mediaPlayerLogger())
    , m_logIdentifier(m_player->mediaPlayerLogIdentifier())
#endif
{
#if !RELEASE_LOG_DISABLED
    client.setLogIdentifier(m_player->mediaPlayerLogIdentifier());
#endif
}

MockMediaSourcePrivate::~MockMediaSourcePrivate() = default;

MediaSourcePrivate::AddStatus MockMediaSourcePrivate::addSourceBuffer(const ContentType& contentType, RefPtr<SourceBufferPrivate>& outPrivate)
{
    MediaEngineSupportParameters parameters;
    parameters.isMediaSource = true;
    parameters.type = contentType;
    if (MockMediaPlayerMediaSource::supportsType(parameters) == MediaPlayer::SupportsType::IsNotSupported)
        return AddStatus::NotSupported;

    m_sourceBuffers.append(MockSourceBufferPrivate::create(*this));
    outPrivate = m_sourceBuffers.last();
    outPrivate->setMediaSourceDuration(duration());

    return AddStatus::Ok;
}

RefPtr<MediaPlayerPrivateInterface> MockMediaSourcePrivate::player() const
{
    return m_player.get();
}

void MockMediaSourcePrivate::setPlayer(MediaPlayerPrivateInterface* player)
{
    m_player = downcast<MockMediaPlayerMediaSource>(player);
}

void MockMediaSourcePrivate::durationChanged(const MediaTime& duration)
{
    MediaSourcePrivate::durationChanged(duration);
    if (m_player)
        m_player->updateDuration(duration);
}

void MockMediaSourcePrivate::markEndOfStream(EndOfStreamStatus status)
{
    if (m_player && status == EndOfStreamStatus::NoError)
        m_player->setNetworkState(MediaPlayer::NetworkState::Loaded);
    MediaSourcePrivate::markEndOfStream(status);
}

MediaPlayer::ReadyState MockMediaSourcePrivate::mediaPlayerReadyState() const
{
    if (m_player)
        return m_player->readyState();
    return MediaPlayer::ReadyState::HaveNothing;
}

void MockMediaSourcePrivate::setMediaPlayerReadyState(MediaPlayer::ReadyState readyState)
{
    if (m_player)
        m_player->setReadyState(readyState);
}

void MockMediaSourcePrivate::notifyActiveSourceBuffersChanged()
{
    if (m_player)
        m_player->notifyActiveSourceBuffersChanged();
}

std::optional<VideoPlaybackQualityMetrics> MockMediaSourcePrivate::videoPlaybackQualityMetrics()
{
    return VideoPlaybackQualityMetrics {
        m_totalVideoFrames,
        m_droppedVideoFrames,
        m_corruptedVideoFrames,
        m_totalFrameDelay.toDouble(),
        0,
    };
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& MockMediaSourcePrivate::logChannel() const
{
    return LogMediaSource;
}
#endif

}

#endif
