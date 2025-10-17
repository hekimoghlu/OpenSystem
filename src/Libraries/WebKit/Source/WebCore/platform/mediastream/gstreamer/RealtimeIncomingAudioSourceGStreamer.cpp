/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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

#if USE(GSTREAMER_WEBRTC)
#include "RealtimeIncomingAudioSourceGStreamer.h"

#include "GStreamerAudioData.h"
#include "GStreamerAudioStreamDescription.h"
#include <wtf/text/MakeString.h>

GST_DEBUG_CATEGORY(webkit_webrtc_incoming_audio_debug);
#define GST_CAT_DEFAULT webkit_webrtc_incoming_audio_debug

namespace WebCore {

RealtimeIncomingAudioSourceGStreamer::RealtimeIncomingAudioSourceGStreamer(AtomString&& audioTrackId)
    : RealtimeIncomingSourceGStreamer(CaptureDevice { WTFMove(audioTrackId), CaptureDevice::DeviceType::Microphone, emptyString() })
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_webrtc_incoming_audio_debug, "webkitwebrtcincomingaudio", 0, "WebKit WebRTC incoming audio");
    });
}

RealtimeIncomingAudioSourceGStreamer::~RealtimeIncomingAudioSourceGStreamer()
{
    stop();
}

const RealtimeMediaSourceSettings& RealtimeIncomingAudioSourceGStreamer::settings()
{
    return m_currentSettings;
}

void RealtimeIncomingAudioSourceGStreamer::dispatchSample(GRefPtr<GstSample>&& sample)
{
    ASSERT(isMainThread());
    auto presentationTime = MediaTime(GST_TIME_AS_USECONDS(GST_BUFFER_PTS(gst_sample_get_buffer(sample.get()))), G_USEC_PER_SEC);
    GStreamerAudioStreamDescription description;
    GStreamerAudioData frames(WTFMove(sample), description.getInfo());
    audioSamplesAvailable(presentationTime, frames, description, 0);
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
