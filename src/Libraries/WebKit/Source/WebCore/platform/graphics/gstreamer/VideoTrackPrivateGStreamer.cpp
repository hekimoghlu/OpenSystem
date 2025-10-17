/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "VideoTrackPrivateGStreamer.h"

#include "GStreamerCommon.h"
#include "MediaPlayerPrivateGStreamer.h"
#include "VP9Utilities.h"
#include <gst/pbutils/pbutils.h>
#include <wtf/Scope.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

GST_DEBUG_CATEGORY(webkit_video_track_debug);
#define GST_CAT_DEFAULT webkit_video_track_debug

static void ensureVideoTrackDebugCategoryInitialized()
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_video_track_debug, "webkitvideotrack", 0, "WebKit Video Track");
    });
}

VideoTrackPrivateGStreamer::VideoTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GRefPtr<GstPad>&& pad, bool shouldHandleStreamStartEvent)
    : TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Video, this, index, WTFMove(pad), shouldHandleStreamStartEvent)
    , m_player(WTFMove(player))
{
    ensureVideoTrackDebugCategoryInitialized();
    installUpdateConfigurationHandlers();
}

VideoTrackPrivateGStreamer::VideoTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GRefPtr<GstPad>&& pad, TrackID trackId)
    : TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Video, this, index, WTFMove(pad), trackId)
    , m_player(WTFMove(player))
{
    ensureVideoTrackDebugCategoryInitialized();
    installUpdateConfigurationHandlers();
}

VideoTrackPrivateGStreamer::VideoTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GstStream* stream)
    : TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Video, this, index, stream)
    , m_player(WTFMove(player))
{
    ensureVideoTrackDebugCategoryInitialized();
    installUpdateConfigurationHandlers();

    auto caps = adoptGRef(gst_stream_get_caps(m_stream.get()));
    updateConfigurationFromCaps(WTFMove(caps));

    auto tags = adoptGRef(gst_stream_get_tags(m_stream.get()));
    updateConfigurationFromTags(WTFMove(tags));
}

void VideoTrackPrivateGStreamer::capsChanged(TrackID streamId, GRefPtr<GstCaps>&& caps)
{
    ASSERT(isMainThread());
    updateConfigurationFromCaps(WTFMove(caps));

    RefPtr player = m_player.get();
    if (!player)
        return;

    auto codec = player->codecForStreamId(streamId);
    if (codec.isEmpty())
        return;

    auto configuration = this->configuration();
    GST_DEBUG_OBJECT(objectForLogging(), "Setting codec to %s", codec.ascii().data());
    configuration.codec = WTFMove(codec);
    setConfiguration(WTFMove(configuration));
}

void VideoTrackPrivateGStreamer::updateConfigurationFromTags(GRefPtr<GstTagList>&& tags)
{
    ASSERT(isMainThread());
    GST_DEBUG_OBJECT(objectForLogging(), "Updating video configuration from %" GST_PTR_FORMAT, tags.get());
    if (!tags)
        return;

    if (updateTrackIDFromTags(tags)) {
        GST_DEBUG_OBJECT(objectForLogging(), "Video track ID set from container-specific-track-id tag %" G_GUINT64_FORMAT, *m_trackID);
        notifyClients([trackID = *m_trackID](auto& client) {
            client.idChanged(trackID);
        });
    }

    unsigned bitrate;
    if (!gst_tag_list_get_uint(tags.get(), GST_TAG_BITRATE, &bitrate))
        return;

    GST_DEBUG_OBJECT(objectForLogging(), "Setting bitrate to %u", bitrate);
    auto configuration = this->configuration();
    configuration.bitrate = bitrate;
    setConfiguration(WTFMove(configuration));
}

void VideoTrackPrivateGStreamer::updateConfigurationFromCaps(GRefPtr<GstCaps>&& caps)
{
    ASSERT(isMainThread());
    if (!caps || !gst_caps_is_fixed(caps.get()))
        return;

    GST_DEBUG_OBJECT(objectForLogging(), "Updating video configuration from %" GST_PTR_FORMAT, caps.get());
    auto configuration = this->configuration();
    auto scopeExit = makeScopeExit([&] {
        setConfiguration(WTFMove(configuration));
    });

#if GST_CHECK_VERSION(1, 20, 0)
    GUniquePtr<char> mimeCodec(gst_codec_utils_caps_get_mime_codec(caps.get()));
    if (mimeCodec) {
        String codec = unsafeSpan(mimeCodec.get());
        if (!webkitGstCheckVersion(1, 22, 8)) {
            // The gst_codec_utils_caps_get_mime_codec() function will return all the codec parameters,
            // including the default ones, so to strip them away, re-parse the returned string, using
            // WebCore VPx codec string parser.
            // This workaround is needed only for GStreamer versions older than 1.22.8. Merge-request:
            // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/merge_requests/5716
            if (codec.startsWith("vp09"_s) && codec.endsWith(".01.01.01.01.00"_s)) {
                auto parsedRecord = parseVPCodecParameters(codec);
                ASSERT(parsedRecord);
                codec = createVPCodecParametersString(*parsedRecord);
            }
        }
        configuration.codec = WTFMove(codec);
    }
#endif

    if (areEncryptedCaps(caps.get())) {
        if (auto videoResolution = getVideoResolutionFromCaps(caps.get())) {
            configuration.width = videoResolution->width();
            configuration.height = videoResolution->height();
        }
        return;
    }

    GstVideoInfo info;
    if (gst_video_info_from_caps(&info, caps.get())) {
        if (GST_VIDEO_INFO_FPS_N(&info))
            gst_util_fraction_to_double(GST_VIDEO_INFO_FPS_N(&info), GST_VIDEO_INFO_FPS_D(&info), &configuration.framerate);

        configuration.width = GST_VIDEO_INFO_WIDTH(&info);
        configuration.height = GST_VIDEO_INFO_HEIGHT(&info);
        configuration.colorSpace = videoColorSpaceFromInfo(info);
    }
}

VideoTrackPrivate::Kind VideoTrackPrivateGStreamer::kind() const
{
    if (m_stream && gst_stream_get_stream_flags(m_stream.get()) & GST_STREAM_FLAG_SELECT)
        return VideoTrackPrivate::Kind::Main;

    return VideoTrackPrivate::kind();
}

void VideoTrackPrivateGStreamer::disconnect()
{
    m_taskQueue.startAborting();

    if (m_stream)
        g_signal_handlers_disconnect_matched(m_stream.get(), G_SIGNAL_MATCH_DATA, 0, 0, nullptr, nullptr, this);

    m_player = nullptr;
    TrackPrivateBaseGStreamer::disconnect();

    m_taskQueue.finishAborting();
}

void VideoTrackPrivateGStreamer::setSelected(bool selected)
{
    if (selected == this->selected())
        return;
    VideoTrackPrivate::setSelected(selected);

    RefPtr player = m_player.get();
    if (player)
        player->updateEnabledVideoTrack();
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
