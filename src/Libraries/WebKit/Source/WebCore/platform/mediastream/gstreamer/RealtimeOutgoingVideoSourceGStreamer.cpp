/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
#include "RealtimeOutgoingVideoSourceGStreamer.h"

#if USE(GSTREAMER_WEBRTC)

#include "GStreamerCommon.h"
#include "GStreamerMediaStreamSource.h"
#include "GStreamerRegistryScanner.h"
#include "GStreamerVideoRTPPacketizer.h"
#include "MediaStreamTrack.h"
#include <wtf/text/MakeString.h>

GST_DEBUG_CATEGORY(webkit_webrtc_outgoing_video_debug);
#define GST_CAT_DEFAULT webkit_webrtc_outgoing_video_debug

namespace WebCore {

RealtimeOutgoingVideoSourceGStreamer::RealtimeOutgoingVideoSourceGStreamer(const RefPtr<UniqueSSRCGenerator>& ssrcGenerator, const String& mediaStreamId, MediaStreamTrack& track)
    : RealtimeOutgoingMediaSourceGStreamer(RealtimeOutgoingMediaSourceGStreamer::Type::Video, ssrcGenerator, mediaStreamId, track)
{
    initializePreProcessor();
}

RealtimeOutgoingVideoSourceGStreamer::RealtimeOutgoingVideoSourceGStreamer(const RefPtr<UniqueSSRCGenerator>& ssrcGenerator)
    : RealtimeOutgoingMediaSourceGStreamer(RealtimeOutgoingMediaSourceGStreamer::Type::Video, ssrcGenerator)
{
    initializePreProcessor();

    m_outgoingSource = gst_element_factory_make("videotestsrc", nullptr);
    gst_util_set_object_arg(G_OBJECT(m_outgoingSource.get()), "pattern", "black");
    g_object_set(m_outgoingSource.get(), "is-live", TRUE, "do-timestamp", TRUE, nullptr);
    gst_bin_add(GST_BIN_CAST(m_bin.get()), m_outgoingSource.get());
}

RealtimeOutgoingVideoSourceGStreamer::~RealtimeOutgoingVideoSourceGStreamer() = default;

void RealtimeOutgoingVideoSourceGStreamer::initializePreProcessor()
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_webrtc_outgoing_video_debug, "webkitwebrtcoutgoingvideo", 0, "WebKit WebRTC outgoing video");
    });
    registerWebKitGStreamerElements();

    static Atomic<uint64_t> sourceCounter = 0;
    gst_element_set_name(m_bin.get(), makeString("outgoing-video-source-"_s, sourceCounter.exchangeAdd(1)).ascii().data());

    m_preProcessor = gst_bin_new(nullptr);

    auto videoConvert = makeGStreamerElement("videoconvert", nullptr);

    auto videoFlip = makeGStreamerElement("autovideoflip", nullptr);
    if (!videoFlip) {
        GST_DEBUG("autovideoflip element not available, falling back to videoflip");
        videoFlip = makeGStreamerElement("videoflip", nullptr);
    }
    gst_util_set_object_arg(G_OBJECT(videoFlip), "video-direction", "auto");

    gst_bin_add_many(GST_BIN_CAST(m_preProcessor.get()), videoFlip, videoConvert, nullptr);
    gst_element_link(videoFlip, videoConvert);

    if (auto pad = adoptGRef(gst_bin_find_unlinked_pad(GST_BIN_CAST(m_preProcessor.get()), GST_PAD_SRC)))
        gst_element_add_pad(GST_ELEMENT_CAST(m_preProcessor.get()), gst_ghost_pad_new("src", pad.get()));
    if (auto pad = adoptGRef(gst_bin_find_unlinked_pad(GST_BIN_CAST(m_preProcessor.get()), GST_PAD_SINK)))
        gst_element_add_pad(GST_ELEMENT_CAST(m_preProcessor.get()), gst_ghost_pad_new("sink", pad.get()));

    gst_bin_add(GST_BIN_CAST(m_bin.get()), m_preProcessor.get());
}

void RealtimeOutgoingVideoSourceGStreamer::teardown()
{
    RealtimeOutgoingMediaSourceGStreamer::teardown();
    m_preProcessor.clear();
}

RTCRtpCapabilities RealtimeOutgoingVideoSourceGStreamer::rtpCapabilities() const
{
    auto& registryScanner = GStreamerRegistryScanner::singleton();
    return registryScanner.videoRtpCapabilities(GStreamerRegistryScanner::Configuration::Encoding);
}

GRefPtr<GstPad> RealtimeOutgoingVideoSourceGStreamer::outgoingSourcePad() const
{
    if (WEBKIT_IS_MEDIA_STREAM_SRC(m_outgoingSource.get()))
        return adoptGRef(gst_element_get_static_pad(m_outgoingSource.get(), "video_src0"));
    return adoptGRef(gst_element_get_static_pad(m_outgoingSource.get(), "src"));
}

RefPtr<GStreamerRTPPacketizer> RealtimeOutgoingVideoSourceGStreamer::createPacketizer(RefPtr<UniqueSSRCGenerator> ssrcGenerator, const GstStructure* codecParameters, GUniquePtr<GstStructure>&& encodingParameters)
{
    return GStreamerVideoRTPPacketizer::create(ssrcGenerator, codecParameters, WTFMove(encodingParameters));
}

void RealtimeOutgoingVideoSourceGStreamer::dispatchBitrateRequest(uint32_t bitrate)
{
    gst_element_send_event(m_bin.get(), gst_event_new_custom(GST_EVENT_CUSTOM_DOWNSTREAM_OOB, gst_structure_new("encoder-bitrate-change-request", "bitrate", G_TYPE_UINT, static_cast<uint32_t>(bitrate / 1000), nullptr)));
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
