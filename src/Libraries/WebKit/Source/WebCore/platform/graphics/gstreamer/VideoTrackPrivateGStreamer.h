/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "MediaPlayerPrivateGStreamer.h"
#include "TrackPrivateBaseGStreamer.h"
#include "VideoTrackPrivate.h"

#include <wtf/ThreadSafeWeakPtr.h>

namespace WebCore {
class MediaPlayerPrivateGStreamer;

class VideoTrackPrivateGStreamer final : public VideoTrackPrivate, public TrackPrivateBaseGStreamer {
public:
    static Ref<VideoTrackPrivateGStreamer> create(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GRefPtr<GstPad>&& pad, bool shouldHandleStreamStartEvent = true)
    {
        return adoptRef(*new VideoTrackPrivateGStreamer(WTFMove(player), index, WTFMove(pad), shouldHandleStreamStartEvent));
    }

    static Ref<VideoTrackPrivateGStreamer> create(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GRefPtr<GstPad>&& pad, TrackID trackId)
    {
        return adoptRef(*new VideoTrackPrivateGStreamer(WTFMove(player), index, WTFMove(pad), trackId));
    }

    static Ref<VideoTrackPrivateGStreamer> create(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GstStream* stream)
    {
        return adoptRef(*new VideoTrackPrivateGStreamer(WTFMove(player), index, stream));
    }

    Kind kind() const final;

    void disconnect() final;

    void setSelected(bool) final;
    void setActive(bool enabled) final { setSelected(enabled); }

    int trackIndex() const final { return m_index; }

    TrackID id() const final { return m_trackID.value_or(m_id); }
    std::optional<AtomString> trackUID() const final
    {
        auto player = m_player.get();

        if (player && player->isMediaStreamPlayer())
            return m_gstStreamId;

        return std::nullopt;
    }

    AtomString label() const final { return m_label; }
    AtomString language() const final { return m_language; }

    void updateConfigurationFromCaps(GRefPtr<GstCaps>&&) final;

protected:
    void updateConfigurationFromTags(GRefPtr<GstTagList>&&) final;

    void tagsChanged(GRefPtr<GstTagList>&& tags) final { updateConfigurationFromTags(WTFMove(tags)); }
    void capsChanged(TrackID, GRefPtr<GstCaps>&&) final;

private:
    VideoTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&&, unsigned index, GRefPtr<GstPad>&&, bool shouldHandleStreamStartEvent);
    VideoTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&&, unsigned index, GRefPtr<GstPad>&&, TrackID);
    VideoTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&&, unsigned index, GstStream*);

    ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer> m_player;
};

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
