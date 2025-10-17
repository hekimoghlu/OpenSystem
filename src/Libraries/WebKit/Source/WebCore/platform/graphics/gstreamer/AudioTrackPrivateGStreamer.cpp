/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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

#include "AudioTrackPrivateGStreamer.h"

#include "GStreamerCommon.h"
#include "MediaPlayerPrivateGStreamer.h"
#include <gst/pbutils/pbutils.h>
#include <wtf/Scope.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebCore {

GST_DEBUG_CATEGORY(webkit_audio_track_debug);
#define GST_CAT_DEFAULT webkit_audio_track_debug

static void ensureDebugCategoryInitialized()
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_audio_track_debug, "webkitaudiotrack", 0, "WebKit Audio Track");
    });
}

AudioTrackPrivateGStreamer::AudioTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GRefPtr<GstPad>&& pad, bool shouldHandleStreamStartEvent)
    : TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Audio, this, index, WTFMove(pad), shouldHandleStreamStartEvent)
    , m_player(WTFMove(player))
{
    ensureDebugCategoryInitialized();
    installUpdateConfigurationHandlers();
}

AudioTrackPrivateGStreamer::AudioTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GRefPtr<GstPad>&& pad, TrackID trackId)
    : TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Audio, this, index, WTFMove(pad), trackId)
    , m_player(WTFMove(player))
{
    ensureDebugCategoryInitialized();
    installUpdateConfigurationHandlers();
}

AudioTrackPrivateGStreamer::AudioTrackPrivateGStreamer(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player, unsigned index, GstStream* stream)
    : TrackPrivateBaseGStreamer(TrackPrivateBaseGStreamer::TrackType::Audio, this, index, stream)
    , m_player(WTFMove(player))
{
    ensureDebugCategoryInitialized();
    installUpdateConfigurationHandlers();

    auto caps = adoptGRef(gst_stream_get_caps(m_stream.get()));
    updateConfigurationFromCaps(WTFMove(caps));

    auto tags = adoptGRef(gst_stream_get_tags(m_stream.get()));
    updateConfigurationFromTags(WTFMove(tags));
}

void AudioTrackPrivateGStreamer::capsChanged(TrackID streamId, GRefPtr<GstCaps>&& caps)
{
    ASSERT(isMainThread());
    updateConfigurationFromCaps(WTFMove(caps));

    RefPtr player = m_player.get();
    if (!player)
        return;

    auto codec = player->codecForStreamId(streamId);
    if (codec.isEmpty())
        return;

    GST_DEBUG_OBJECT(objectForLogging(), "Setting codec to %s", codec.ascii().data());
    auto configuration = this->configuration();
    configuration.codec = WTFMove(codec);
    setConfiguration(WTFMove(configuration));
}

void AudioTrackPrivateGStreamer::updateConfigurationFromTags(GRefPtr<GstTagList>&& tags)
{
    ASSERT(isMainThread());
    GST_DEBUG_OBJECT(objectForLogging(), "Updating audio configuration from %" GST_PTR_FORMAT, tags.get());
    if (!tags)
        return;

    if (updateTrackIDFromTags(tags)) {
        GST_DEBUG_OBJECT(objectForLogging(), "Audio track ID set from container-specific-track-id tag %" G_GUINT64_FORMAT, *m_trackID);
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

void AudioTrackPrivateGStreamer::updateConfigurationFromCaps(GRefPtr<GstCaps>&& caps)
{
    ASSERT(isMainThread());
    if (!caps || !gst_caps_is_fixed(caps.get()))
        return;

    GST_DEBUG_OBJECT(objectForLogging(), "Updating audio configuration from %" GST_PTR_FORMAT, caps.get());
    auto configuration = this->configuration();
    auto scopeExit = makeScopeExit([&] {
        setConfiguration(WTFMove(configuration));
    });

#if GST_CHECK_VERSION(1, 20, 0)
    GUniquePtr<char> mimeCodec(gst_codec_utils_caps_get_mime_codec(caps.get()));
    if (mimeCodec)
        configuration.codec = unsafeSpan(mimeCodec.get());
#endif

    if (areEncryptedCaps(caps.get())) {
        const auto* structure = gst_caps_get_structure(caps.get(), 0);
        if (auto sampleRate = gstStructureGet<int>(structure, "rate"_s))
            configuration.sampleRate = *sampleRate;
        if (auto numberOfChannels = gstStructureGet<int>(structure, "channels"_s))
            configuration.numberOfChannels = *numberOfChannels;
        return;
    }

    GstAudioInfo info;
    if (gst_audio_info_from_caps(&info, caps.get())) {
        configuration.sampleRate = GST_AUDIO_INFO_RATE(&info);
        configuration.numberOfChannels = GST_AUDIO_INFO_CHANNELS(&info);
    }
}

AudioTrackPrivate::Kind AudioTrackPrivateGStreamer::kind() const
{
    if (m_stream && gst_stream_get_stream_flags(m_stream.get()) & GST_STREAM_FLAG_SELECT)
        return AudioTrackPrivate::Kind::Main;

    return AudioTrackPrivate::kind();
}

void AudioTrackPrivateGStreamer::disconnect()
{
    m_taskQueue.startAborting();

    if (m_stream)
        g_signal_handlers_disconnect_matched(m_stream.get(), G_SIGNAL_MATCH_DATA, 0, 0, nullptr, nullptr, this);

    m_player = nullptr;
    TrackPrivateBaseGStreamer::disconnect();

    m_taskQueue.finishAborting();
}

void AudioTrackPrivateGStreamer::setEnabled(bool enabled)
{
    if (enabled == this->enabled())
        return;
    AudioTrackPrivate::setEnabled(enabled);

    RefPtr player = m_player.get();
    if (player)
        player->updateEnabledAudioTrack();
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
