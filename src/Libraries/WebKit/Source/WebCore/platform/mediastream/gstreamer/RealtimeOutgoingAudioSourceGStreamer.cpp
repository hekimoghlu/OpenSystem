/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#include "RealtimeOutgoingAudioSourceGStreamer.h"

#if USE(GSTREAMER_WEBRTC)

#include "GStreamerAudioRTPPacketizer.h"
#include "GStreamerCommon.h"
#include "GStreamerMediaStreamSource.h"
#include "GStreamerRegistryScanner.h"
#include "MediaStreamTrack.h"
#include "NotImplemented.h"
#include <wtf/text/MakeString.h>

GST_DEBUG_CATEGORY(webkit_webrtc_outgoing_audio_debug);
#define GST_CAT_DEFAULT webkit_webrtc_outgoing_audio_debug

namespace WebCore {

RealtimeOutgoingAudioSourceGStreamer::RealtimeOutgoingAudioSourceGStreamer(const RefPtr<UniqueSSRCGenerator>& ssrcGenerator, const String& mediaStreamId, MediaStreamTrack& track)
    : RealtimeOutgoingMediaSourceGStreamer(RealtimeOutgoingMediaSourceGStreamer::Type::Audio, ssrcGenerator, mediaStreamId, track)
{
    initialize();
}

RealtimeOutgoingAudioSourceGStreamer::RealtimeOutgoingAudioSourceGStreamer(const RefPtr<UniqueSSRCGenerator>& ssrcGenerator)
    : RealtimeOutgoingMediaSourceGStreamer(RealtimeOutgoingMediaSourceGStreamer::Type::Audio, ssrcGenerator)
{
    initialize();

    m_outgoingSource = gst_element_factory_make("audiotestsrc", nullptr);
    gst_util_set_object_arg(G_OBJECT(m_outgoingSource.get()), "wave", "silence");
    g_object_set(m_outgoingSource.get(), "is-live", TRUE, "do-timestamp", TRUE, nullptr);
    gst_bin_add(GST_BIN_CAST(m_bin.get()), m_outgoingSource.get());
}

RealtimeOutgoingAudioSourceGStreamer::~RealtimeOutgoingAudioSourceGStreamer() = default;

void RealtimeOutgoingAudioSourceGStreamer::initialize()
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_webrtc_outgoing_audio_debug, "webkitwebrtcoutgoingaudio", 0, "WebKit WebRTC outgoing audio");
    });
    static Atomic<uint64_t> sourceCounter = 0;
    gst_element_set_name(m_bin.get(), makeString("outgoing-audio-source-"_s, sourceCounter.exchangeAdd(1)).ascii().data());
}

RTCRtpCapabilities RealtimeOutgoingAudioSourceGStreamer::rtpCapabilities() const
{
    auto& registryScanner = GStreamerRegistryScanner::singleton();
    return registryScanner.audioRtpCapabilities(GStreamerRegistryScanner::Configuration::Encoding);
}

GRefPtr<GstPad> RealtimeOutgoingAudioSourceGStreamer::outgoingSourcePad() const
{
    if (WEBKIT_IS_MEDIA_STREAM_SRC(m_outgoingSource.get()))
        return adoptGRef(gst_element_get_static_pad(m_outgoingSource.get(), "audio_src0"));
    return adoptGRef(gst_element_get_static_pad(m_outgoingSource.get(), "src"));
}

RefPtr<GStreamerRTPPacketizer> RealtimeOutgoingAudioSourceGStreamer::createPacketizer(RefPtr<UniqueSSRCGenerator> ssrcGenerator, const GstStructure* codecParameters, GUniquePtr<GstStructure>&& encodingParameters)
{
    return GStreamerAudioRTPPacketizer::create(ssrcGenerator, codecParameters, WTFMove(encodingParameters));
}

void RealtimeOutgoingAudioSourceGStreamer::dispatchBitrateRequest(uint32_t)
{
    notImplemented();
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
