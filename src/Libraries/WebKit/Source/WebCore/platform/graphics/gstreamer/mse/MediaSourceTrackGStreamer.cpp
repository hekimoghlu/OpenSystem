/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#include "MediaSourceTrackGStreamer.h"

#if ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)

GST_DEBUG_CATEGORY_STATIC(webkit_mse_track_debug);
#define GST_CAT_DEFAULT webkit_mse_track_debug

namespace WebCore {

MediaSourceTrackGStreamer::MediaSourceTrackGStreamer(TrackPrivateBaseGStreamer::TrackType type, TrackID trackId, GRefPtr<GstCaps>&& initialCaps)
    : m_type(type)
    , m_id(trackId)
    , m_initialCaps(WTFMove(initialCaps))
    , m_queueDataMutex(trackId)
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_mse_track_debug, "webkitmsetrack", 0, "WebKit MSE Track");
    });
}

MediaSourceTrackGStreamer::~MediaSourceTrackGStreamer()
{
    ASSERT(m_isRemoved);
}

Ref<MediaSourceTrackGStreamer> MediaSourceTrackGStreamer::create(TrackPrivateBaseGStreamer::TrackType type, TrackID trackId, GRefPtr<GstCaps>&& initialCaps)
{
    return adoptRef(*new MediaSourceTrackGStreamer(type, trackId, WTFMove(initialCaps)));
}

bool MediaSourceTrackGStreamer::isReadyForMoreSamples()
{
    ASSERT(isMainThread());
    DataMutexLocker queue { m_queueDataMutex };
    return !queue->isFull();
}

void MediaSourceTrackGStreamer::notifyWhenReadyForMoreSamples(TrackQueue::LowLevelHandler&& handler)
{
    ASSERT(isMainThread());
    DataMutexLocker queue { m_queueDataMutex };
    queue->notifyWhenLowLevel(WTFMove(handler));
}

void MediaSourceTrackGStreamer::enqueueObject(GRefPtr<GstMiniObject>&& object)
{
    ASSERT(isMainThread());
    DataMutexLocker queue { m_queueDataMutex };
    queue->enqueueObject(WTFMove(object));
}

void MediaSourceTrackGStreamer::clearQueue()
{
    ASSERT(isMainThread());
    DataMutexLocker queue { m_queueDataMutex };
    queue->clear();
}

void MediaSourceTrackGStreamer::remove()
{
    ASSERT(isMainThread());
    m_isRemoved = true;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)
