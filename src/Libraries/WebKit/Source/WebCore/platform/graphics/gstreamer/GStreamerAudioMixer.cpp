/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 27, 2023.
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
#include "GStreamerAudioMixer.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <wtf/NeverDestroyed.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_media_gst_audio_mixer_debug);
#define GST_CAT_DEFAULT webkit_media_gst_audio_mixer_debug

bool GStreamerAudioMixer::isAvailable()
{
    return isGStreamerPluginAvailable("inter") && isGStreamerPluginAvailable("audiomixer");
}

GStreamerAudioMixer& GStreamerAudioMixer::singleton()
{
    static NeverDestroyed<GStreamerAudioMixer> sharedInstance;
    return sharedInstance;
}

GStreamerAudioMixer::GStreamerAudioMixer()
{
    GST_DEBUG_CATEGORY_INIT(webkit_media_gst_audio_mixer_debug, "webkitaudiomixer", 0, "WebKit GStreamer audio mixer");
    m_pipeline = gst_element_factory_make("pipeline", "webkitaudiomixer");
    registerActivePipeline(m_pipeline);
    connectSimpleBusMessageCallback(m_pipeline.get());

    m_mixer = makeGStreamerElement("audiomixer", nullptr);
    auto* audioSink = createAutoAudioSink({ });

    gst_bin_add_many(GST_BIN_CAST(m_pipeline.get()), m_mixer.get(), audioSink, nullptr);
    gst_element_link(m_mixer.get(), audioSink);
    gst_element_set_state(m_pipeline.get(), GST_STATE_READY);
}

void GStreamerAudioMixer::ensureState(GstStateChange stateChange)
{
    GST_DEBUG_OBJECT(m_pipeline.get(), "Handling %s transition (%u mixer pads)", gst_state_change_get_name(stateChange), m_mixer->numsinkpads);

    switch (stateChange) {
    case GST_STATE_CHANGE_READY_TO_PAUSED:
        gst_element_set_state(m_pipeline.get(), GST_STATE_PAUSED);
        break;
    case GST_STATE_CHANGE_PAUSED_TO_PLAYING:
        gst_element_set_state(m_pipeline.get(), GST_STATE_PLAYING);
        break;
    case GST_STATE_CHANGE_PLAYING_TO_PAUSED:
        if (m_mixer->numsinkpads == 1)
            gst_element_set_state(m_pipeline.get(), GST_STATE_PAUSED);
        break;
    case GST_STATE_CHANGE_PAUSED_TO_READY:
        if (m_mixer->numsinkpads == 1)
            gst_element_set_state(m_pipeline.get(), GST_STATE_READY);
        break;
    case GST_STATE_CHANGE_READY_TO_NULL:
        if (m_mixer->numsinkpads == 1) {
            unregisterPipeline(m_pipeline);
            gst_element_set_state(m_pipeline.get(), GST_STATE_NULL);
        }
        break;
    default:
        break;
    }
}

GRefPtr<GstPad> GStreamerAudioMixer::registerProducer(GstElement* interaudioSink)
{
    GstElement* src = makeGStreamerElement("interaudiosrc", nullptr);
    g_object_set(src, "channel", GST_ELEMENT_NAME(interaudioSink), nullptr);
    g_object_set(interaudioSink, "channel", GST_ELEMENT_NAME(interaudioSink), nullptr);

    auto bin = gst_bin_new(nullptr);
    auto audioResample = makeGStreamerElement("audioresample", nullptr);
    auto audioConvert = makeGStreamerElement("audioconvert", nullptr);
    gst_bin_add_many(GST_BIN_CAST(bin), audioResample, audioConvert, nullptr);
    gst_element_link(audioConvert, audioResample);

    if (auto pad = adoptGRef(gst_bin_find_unlinked_pad(GST_BIN_CAST(bin), GST_PAD_SRC)))
        gst_element_add_pad(GST_ELEMENT_CAST(bin), gst_ghost_pad_new("src", pad.get()));
    if (auto pad = adoptGRef(gst_bin_find_unlinked_pad(GST_BIN_CAST(bin), GST_PAD_SINK)))
        gst_element_add_pad(GST_ELEMENT_CAST(bin), gst_ghost_pad_new("sink", pad.get()));

    gst_bin_add_many(GST_BIN_CAST(m_pipeline.get()), src, bin, nullptr);
    gst_element_link(src, bin);

    bool shouldStart = !m_mixer->numsinkpads;

    auto mixerPad = adoptGRef(gst_element_request_pad_simple(m_mixer.get(), "sink_%u"));
    auto srcPad = adoptGRef(gst_element_get_static_pad(bin, "src"));
    gst_pad_link(srcPad.get(), mixerPad.get());

    if (shouldStart)
        gst_element_set_state(m_pipeline.get(), GST_STATE_READY);
    else
        gst_bin_sync_children_states(GST_BIN_CAST(m_pipeline.get()));

    GST_DEBUG_OBJECT(m_pipeline.get(), "Registered audio producer %" GST_PTR_FORMAT, mixerPad.get());
    GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN_CAST(m_pipeline.get()), GST_DEBUG_GRAPH_SHOW_ALL, "audio-mixer-after-producer-registration");
    return mixerPad;
}

void GStreamerAudioMixer::unregisterProducer(const GRefPtr<GstPad>& mixerPad)
{
    GST_DEBUG_OBJECT(m_pipeline.get(), "Unregistering audio producer %" GST_PTR_FORMAT, mixerPad.get());

    auto peer = adoptGRef(gst_pad_get_peer(mixerPad.get()));
    auto bin = adoptGRef(gst_pad_get_parent_element(peer.get()));
    auto sinkPad = adoptGRef(gst_element_get_static_pad(bin.get(), "sink"));
    auto srcPad = adoptGRef(gst_pad_get_peer(sinkPad.get()));
    auto interaudioSrc = adoptGRef(gst_pad_get_parent_element(srcPad.get()));
    GST_LOG_OBJECT(m_pipeline.get(), "interaudiosrc: %" GST_PTR_FORMAT, interaudioSrc.get());

    gst_element_set_locked_state(interaudioSrc.get(), true);
    gst_element_set_state(interaudioSrc.get(), GST_STATE_NULL);
    gst_element_set_state(bin.get(), GST_STATE_NULL);

    gst_pad_unlink(peer.get(), mixerPad.get());
    gst_element_unlink(interaudioSrc.get(), bin.get());

    gst_element_release_request_pad(m_mixer.get(), mixerPad.get());

    gst_bin_remove_many(GST_BIN_CAST(m_pipeline.get()), interaudioSrc.get(), bin.get(), nullptr);

    if (!m_mixer->numsinkpads)
        gst_element_set_state(m_pipeline.get(), GST_STATE_NULL);

    GST_DEBUG_BIN_TO_DOT_FILE_WITH_TS(GST_BIN_CAST(m_pipeline.get()), GST_DEBUG_GRAPH_SHOW_ALL, "audio-mixer-after-producer-unregistration");
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif
