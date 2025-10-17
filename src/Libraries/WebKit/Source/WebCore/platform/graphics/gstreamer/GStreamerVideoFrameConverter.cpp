/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
#include "GStreamerVideoFrameConverter.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gl/gl.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/Scope.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_gst_video_frame_converter_debug);
#define GST_CAT_DEFAULT webkit_gst_video_frame_converter_debug

GStreamerVideoFrameConverter& GStreamerVideoFrameConverter::singleton()
{
    static NeverDestroyed<GStreamerVideoFrameConverter> sharedInstance;
    return sharedInstance;
}

GStreamerVideoFrameConverter::GStreamerVideoFrameConverter()
{
    ensureGStreamerInitialized();
    GST_DEBUG_CATEGORY_INIT(webkit_gst_video_frame_converter_debug, "webkitvideoframeconverter", 0, "WebKit GStreamer Video Frame Converter");
    m_pipeline = gst_element_factory_make("pipeline", "video-frame-converter");
    m_src = makeGStreamerElement("appsrc", nullptr);
    auto gldownload = makeGStreamerElement("gldownload", nullptr);
    auto videoconvert = makeGStreamerElement("videoconvert", nullptr);
    auto videoscale = makeGStreamerElement("videoscale", nullptr);
    m_sink = makeGStreamerElement("appsink", nullptr);
    g_object_set(m_sink.get(), "enable-last-sample", FALSE, "max-buffers", 1, nullptr);

    gst_bin_add_many(GST_BIN_CAST(m_pipeline.get()), m_src.get(), gldownload, videoconvert, videoscale, m_sink.get(), nullptr);
    gst_element_link_many(m_src.get(), gldownload, videoconvert, videoscale, m_sink.get(), nullptr);
}

GRefPtr<GstSample> GStreamerVideoFrameConverter::convert(const GRefPtr<GstSample>& sample, const GRefPtr<GstCaps>& destinationCaps)
{
    auto inputCaps = gst_sample_get_caps(sample.get());
    if (gst_caps_is_equal(inputCaps, destinationCaps.get()))
        return GRefPtr(sample);

    if (!setGstElementGLContext(m_sink.get(), GST_GL_DISPLAY_CONTEXT_TYPE))
        return nullptr;
    if (!setGstElementGLContext(m_sink.get(), "gst.gl.app_context"))
        return nullptr;

    unsigned capsSize = gst_caps_get_size(destinationCaps.get());
    auto newCaps = adoptGRef(gst_caps_new_empty());
    for (unsigned i = 0; i < capsSize; i++) {
        auto structure = gst_caps_get_structure(destinationCaps.get(), i);
        auto modifiedStructure = gst_structure_copy(structure);
        gst_structure_remove_field(modifiedStructure, "framerate");
        gst_caps_append_structure(newCaps.get(), modifiedStructure);
    }

    GST_TRACE_OBJECT(m_pipeline.get(), "Converting sample with caps %" GST_PTR_FORMAT " to %" GST_PTR_FORMAT, inputCaps, newCaps.get());
    g_object_set(m_sink.get(), "caps", newCaps.get(), nullptr);

    auto scopeExit = WTF::makeScopeExit([&] {
        gst_element_set_state(m_pipeline.get(), GST_STATE_NULL);
    });

    gst_element_set_state(m_pipeline.get(), GST_STATE_PAUSED);
    gst_app_src_push_sample(GST_APP_SRC_CAST(m_src.get()), sample.get());

    auto bus = adoptGRef(gst_element_get_bus(m_pipeline.get()));
    auto message = adoptGRef(gst_bus_timed_pop_filtered(bus.get(), 200 * GST_MSECOND, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_ASYNC_DONE)));
    if (!message) {
        GST_ERROR_OBJECT(m_pipeline.get(), "Video frame conversion 200ms timeout expired.");
        return nullptr;
    }

    if (GST_MESSAGE_TYPE(message.get()) == GST_MESSAGE_ERROR) {
        GST_ERROR_OBJECT(m_pipeline.get(), "Unable to convert video frame. Error: %" GST_PTR_FORMAT, message.get());
        return nullptr;
    }

    auto outputSample = adoptGRef(gst_app_sink_pull_preroll(GST_APP_SINK_CAST(m_sink.get())));
    auto convertedSample = adoptGRef(gst_sample_make_writable(outputSample.leakRef()));
    gst_sample_set_caps(convertedSample.get(), destinationCaps.get());

    GRefPtr buffer = gst_sample_get_buffer(convertedSample.get());
    auto writableBuffer = adoptGRef(gst_buffer_make_writable(buffer.leakRef()));

    if (auto meta = gst_buffer_get_video_meta(writableBuffer.get()))
        gst_buffer_remove_meta(writableBuffer.get(), GST_META_CAST(meta));

    if (auto meta = gst_buffer_get_parent_buffer_meta(writableBuffer.get()))
        gst_buffer_remove_meta(writableBuffer.get(), GST_META_CAST(meta));

    auto structure = gst_caps_get_structure(destinationCaps.get(), 0);
    auto width = gstStructureGet<int>(structure, "width"_s);
    auto height = gstStructureGet<int>(structure, "height"_s);
    auto formatStringView = gstStructureGetString(structure, "format"_s);
    if (width && height && !formatStringView.isEmpty()) {
        auto format = gst_video_format_from_string(formatStringView.toStringWithoutCopying().ascii().data());
        gst_buffer_add_video_meta(writableBuffer.get(), GST_VIDEO_FRAME_FLAG_NONE, format, *width, *height);
    }
    gst_sample_set_buffer(convertedSample.get(), writableBuffer.get());

    return convertedSample;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif
