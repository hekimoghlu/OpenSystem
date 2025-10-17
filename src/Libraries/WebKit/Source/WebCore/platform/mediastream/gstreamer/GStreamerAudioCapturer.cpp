/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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

#if ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
#include "GStreamerAudioCapturer.h"

#include <gst/app/gstappsink.h>

namespace WebCore {

GST_DEBUG_CATEGORY(webkit_audio_capturer_debug);
#define GST_CAT_DEFAULT webkit_audio_capturer_debug

static void initializeAudioCapturerDebugCategory()
{
    static std::once_flag debugRegisteredFlag;
    std::call_once(debugRegisteredFlag, [] {
        GST_DEBUG_CATEGORY_INIT(webkit_audio_capturer_debug, "webkitaudiocapturer", 0, "WebKit Audio Capturer");
    });
}

GStreamerAudioCapturer::GStreamerAudioCapturer(GStreamerCaptureDevice&& device)
    : GStreamerCapturer(WTFMove(device), adoptGRef(gst_caps_new_empty_simple("audio/x-raw")))
{
    initializeAudioCapturerDebugCategory();
}

GStreamerAudioCapturer::GStreamerAudioCapturer()
    : GStreamerCapturer("appsrc", adoptGRef(gst_caps_new_empty_simple("audio/x-raw")), CaptureDevice::DeviceType::Microphone)
{
    initializeAudioCapturerDebugCategory();
}

void GStreamerAudioCapturer::setSinkAudioCallback(SinkAudioDataCallback&& callback)
{
    if (m_sinkAudioDataCallback.first)
        g_signal_handler_disconnect(sink(), m_sinkAudioDataCallback.first);

    m_sinkAudioDataCallback.second = WTFMove(callback);
    m_sinkAudioDataCallback.first = g_signal_connect_swapped(sink(), "new-sample", G_CALLBACK(+[](GStreamerAudioCapturer* capturer, GstElement* sink) -> GstFlowReturn {
        auto gstSample = adoptGRef(gst_app_sink_pull_sample(GST_APP_SINK(sink)));
        auto presentationTime = fromGstClockTime(GST_BUFFER_PTS(gst_sample_get_buffer(gstSample.get())));
        capturer->m_sinkAudioDataCallback.second(WTFMove(gstSample), WTFMove(presentationTime));
        return GST_FLOW_OK;
    }), this);
}

GstElement* GStreamerAudioCapturer::createConverter()
{
    auto* bin = gst_bin_new(nullptr);
    auto* audioconvert = makeGStreamerElement("audioconvert", nullptr);
    auto* audioresample = makeGStreamerElement("audioresample", nullptr);
    gst_bin_add_many(GST_BIN_CAST(bin), audioconvert, audioresample, nullptr);
    gst_element_link(audioconvert, audioresample);

#if USE(GSTREAMER_WEBRTC)
    if (auto audioFilter = makeGStreamerElement("audiornnoise", nullptr)) {
        auto audioconvert2 = makeGStreamerElement("audioconvert", nullptr);
        auto audioresample2 = makeGStreamerElement("audioresample", nullptr);
        gst_bin_add_many(GST_BIN_CAST(bin), audioconvert2, audioresample2, audioFilter, nullptr);
        gst_element_link_many(audioconvert2, audioresample2, audioFilter, audioconvert, nullptr);
    }
#endif

    if (auto pad = adoptGRef(gst_bin_find_unlinked_pad(GST_BIN_CAST(bin), GST_PAD_SRC)))
        gst_element_add_pad(GST_ELEMENT_CAST(bin), gst_ghost_pad_new("src", pad.get()));
    if (auto pad = adoptGRef(gst_bin_find_unlinked_pad(GST_BIN_CAST(bin), GST_PAD_SINK)))
        gst_element_add_pad(GST_ELEMENT_CAST(bin), gst_ghost_pad_new("sink", pad.get()));

    return bin;
}

bool GStreamerAudioCapturer::setSampleRate(int sampleRate)
{
    if (sampleRate <= 0) {
        GST_INFO_OBJECT(m_pipeline.get(), "Not forcing sample rate");
        return false;
    }

    GST_INFO_OBJECT(m_pipeline.get(), "Setting SampleRate %d", sampleRate);
    m_caps = adoptGRef(gst_caps_new_simple("audio/x-raw", "rate", G_TYPE_INT, sampleRate, nullptr));

    if (!m_capsfilter)
        return false;

    g_object_set(m_capsfilter.get(), "caps", m_caps.get(), nullptr);
    return true;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // ENABLE(MEDIA_STREAM) && USE(GSTREAMER)
