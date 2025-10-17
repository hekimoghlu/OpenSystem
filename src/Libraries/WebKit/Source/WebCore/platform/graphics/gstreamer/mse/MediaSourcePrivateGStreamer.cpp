/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 27, 2022.
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
#include "MediaSourcePrivateGStreamer.h"

#if ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)

#include "ContentType.h"
#include "Logging.h"
#include "MediaPlayerPrivateGStreamer.h"
#include "MediaPlayerPrivateGStreamerMSE.h"
#include "MediaSourceTrackGStreamer.h"
#include "NotImplemented.h"
#include "SourceBufferPrivateGStreamer.h"
#include "TimeRanges.h"
#include "WebKitMediaSourceGStreamer.h"
#include <wtf/NativePromise.h>
#include <wtf/RefPtr.h>
#include <wtf/glib/GRefPtr.h>

GST_DEBUG_CATEGORY_STATIC(webkit_mse_private_debug);
#define GST_CAT_DEFAULT webkit_mse_private_debug

namespace WebCore {

Ref<MediaSourcePrivateGStreamer> MediaSourcePrivateGStreamer::open(MediaSourcePrivateClient& mediaSource, MediaPlayerPrivateGStreamerMSE& playerPrivate)
{
    auto mediaSourcePrivate = adoptRef(*new MediaSourcePrivateGStreamer(mediaSource, playerPrivate));
    mediaSource.setPrivateAndOpen(mediaSourcePrivate.copyRef());
    return mediaSourcePrivate;
}

MediaSourcePrivateGStreamer::MediaSourcePrivateGStreamer(MediaSourcePrivateClient& mediaSource, MediaPlayerPrivateGStreamerMSE& playerPrivate)
    : MediaSourcePrivate(mediaSource)
    , m_playerPrivate(playerPrivate)
#if !RELEASE_LOG_DISABLED
    , m_logger(playerPrivate.mediaPlayerLogger())
    , m_logIdentifier(playerPrivate.mediaPlayerLogIdentifier())
#endif
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_mse_private_debug, "webkitmseprivate", 0, "WebKit MSE Private");
    });
}

MediaSourcePrivateGStreamer::~MediaSourcePrivateGStreamer()
{
    ALWAYS_LOG(LOGIDENTIFIER);
}

MediaSourcePrivateGStreamer::AddStatus MediaSourcePrivateGStreamer::addSourceBuffer(const ContentType& contentType, RefPtr<SourceBufferPrivate>& sourceBufferPrivate)
{
    DEBUG_LOG(LOGIDENTIFIER, contentType);

    // Once every SourceBuffer has had an initialization segment appended playback starts and it's too late to add new SourceBuffers.
    if (m_hasAllTracks)
        return MediaSourcePrivateGStreamer::AddStatus::ReachedIdLimit;

    if (!SourceBufferPrivateGStreamer::isContentTypeSupported(contentType))
        return MediaSourcePrivateGStreamer::AddStatus::NotSupported;

    m_sourceBuffers.append(SourceBufferPrivateGStreamer::create(*this, contentType));
    sourceBufferPrivate = m_sourceBuffers.last();
    sourceBufferPrivate->setMediaSourceDuration(duration());
    return MediaSourcePrivateGStreamer::AddStatus::Ok;
}

RefPtr<MediaPlayerPrivateInterface> MediaSourcePrivateGStreamer::player() const
{
    return m_playerPrivate.get();
}

void MediaSourcePrivateGStreamer::setPlayer(MediaPlayerPrivateInterface* player)
{
    m_playerPrivate = downcast<MediaPlayerPrivateGStreamerMSE>(player);
}

RefPtr<MediaPlayerPrivateGStreamerMSE> MediaSourcePrivateGStreamer::platformPlayer() const
{
    return m_playerPrivate.get();
}

void MediaSourcePrivateGStreamer::durationChanged(const MediaTime& duration)
{
    ASSERT(isMainThread());

    RefPtr player = platformPlayer();
    if (!player)
        return;
    MediaSourcePrivate::durationChanged(duration);
    GST_TRACE_OBJECT(player->pipeline(), "Duration: %" GST_TIME_FORMAT, GST_TIME_ARGS(toGstClockTime(duration)));
    if (!duration.isValid() || duration.isNegativeInfinite())
        return;

    player->durationChanged();
}

void MediaSourcePrivateGStreamer::markEndOfStream(EndOfStreamStatus endOfStreamStatus)
{
    ASSERT(isMainThread());

    RefPtr player = platformPlayer();
    if (!player)
        return;

#ifndef GST_DISABLE_GST_DEBUG
    const char* statusString = nullptr;
    switch (endOfStreamStatus) {
    case EndOfStreamStatus::NoError:
        statusString = "no-error";
        break;
    case EndOfStreamStatus::DecodeError:
        statusString = "decode-error";
        break;
    case EndOfStreamStatus::NetworkError:
        statusString = "network-error";
        break;
    }
    GST_DEBUG_OBJECT(player->pipeline(), "Marking EOS, status is %s", statusString);
#endif
    if (endOfStreamStatus == EndOfStreamStatus::NoError) {
        player->setNetworkState(MediaPlayer::NetworkState::Loaded);

        auto bufferedRanges = buffered();
        if (!bufferedRanges.length()) {
            GST_DEBUG("EOS with no buffers");
            player->setEosWithNoBuffers(true);
        }
    }
    MediaSourcePrivate::markEndOfStream(endOfStreamStatus);
}

void MediaSourcePrivateGStreamer::unmarkEndOfStream()
{
    ASSERT(isMainThread());
    RefPtr player = platformPlayer();
    if (!player)
        return;

    player->setEosWithNoBuffers(false);
    MediaSourcePrivate::unmarkEndOfStream();
}

MediaPlayer::ReadyState MediaSourcePrivateGStreamer::mediaPlayerReadyState() const
{
    RefPtr player = platformPlayer();
    return player ? player->readyState() : MediaPlayer::ReadyState::HaveNothing;
}

void MediaSourcePrivateGStreamer::setMediaPlayerReadyState(MediaPlayer::ReadyState state)
{
    if (RefPtr player = platformPlayer())
        player->setReadyState(state);
}

void MediaSourcePrivateGStreamer::startPlaybackIfHasAllTracks()
{
    RefPtr player = platformPlayer();
    if (!player)
        return;

    if (m_hasAllTracks) {
        // Already started, nothing to do.
        return;
    }

    for (auto& sourceBuffer : m_sourceBuffers) {
        if (!sourceBuffer->hasReceivedFirstInitializationSegment()) {
            GST_DEBUG_OBJECT(player->pipeline(), "There are still SourceBuffers without an initialization segment, not starting source yet.");
            return;
        }
    }

    GST_DEBUG_OBJECT(player->pipeline(), "All SourceBuffers have an initialization segment, starting source.");
    m_hasAllTracks = true;

    Vector<RefPtr<MediaSourceTrackGStreamer>> tracks;
    for (auto& privateSourceBuffer : m_sourceBuffers) {
        auto sourceBuffer = downcast<SourceBufferPrivateGStreamer>(privateSourceBuffer);
        for (auto& [_, track] : sourceBuffer->tracks())
            tracks.append(track);
    }
    player->startSource(tracks);
}

TrackID MediaSourcePrivateGStreamer::registerTrackId(TrackID preferredId)
{
    ASSERT(isMainThread());
    RefPtr player = platformPlayer();

    if (m_trackIdRegistry.add(preferredId).isNewEntry) {
        if (player)
            GST_DEBUG_OBJECT(player->pipeline(), "Registered new Track ID: %" PRIu64 "", preferredId);
        return preferredId;
    }

    // If the ID is already known, assign one starting at 100 - this helps avoid a snowball effect
    // where each following ID would now need to be offset by 1.
    auto maxRegisteredId = std::max_element(m_trackIdRegistry.begin(), m_trackIdRegistry.end());
    auto assignedId = std::max((TrackID) 100, *maxRegisteredId + 1);

    ASSERT(m_trackIdRegistry.add(assignedId).isNewEntry);
    if (player)
        GST_DEBUG_OBJECT(player->pipeline(), "Registered new Track ID: %" PRIu64 " (preferred ID would have been %" PRIu64 ")", assignedId, preferredId);

    return assignedId;
}

bool MediaSourcePrivateGStreamer::unregisterTrackId(TrackID trackId)
{
    ASSERT(isMainThread());

    bool res = m_trackIdRegistry.remove(trackId);

    if (RefPtr player = this->platformPlayer()) {
        if (res)
            GST_DEBUG_OBJECT(player->pipeline(), "Unregistered Track ID: %" PRIu64 "", trackId);
        else
            GST_WARNING_OBJECT(player->pipeline(), "Failed to unregister unknown Track ID: %" PRIu64 "", trackId);
    }

    return res;
}

void MediaSourcePrivateGStreamer::notifyActiveSourceBuffersChanged()
{
    if (RefPtr player = platformPlayer())
        player->notifyActiveSourceBuffersChanged();
}

void MediaSourcePrivateGStreamer::detach()
{
    m_hasAllTracks = false;
}

#if !RELEASE_LOG_DISABLED
WTFLogChannel& MediaSourcePrivateGStreamer::logChannel() const
{
    return LogMediaSource;
}

#endif

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)
