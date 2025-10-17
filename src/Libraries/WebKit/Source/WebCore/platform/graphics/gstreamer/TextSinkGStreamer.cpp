/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 6, 2024.
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
#include "TextSinkGStreamer.h"

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "MediaPlayerPrivateGStreamer.h"
#include <gst/app/gstappsink.h>
#include <wtf/glib/WTFGType.h>

GST_DEBUG_CATEGORY_STATIC(webkitTextSinkDebug);
#define GST_CAT_DEFAULT webkitTextSinkDebug

using namespace WebCore;

struct _WebKitTextSinkPrivate {
    GRefPtr<GstElement> appSink;
    ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer> mediaPlayerPrivate;
    std::optional<TrackID> streamId;
};

WEBKIT_DEFINE_TYPE_WITH_CODE(WebKitTextSink, webkit_text_sink, GST_TYPE_BIN,
    GST_DEBUG_CATEGORY_INIT(webkitTextSinkDebug, "webkittextsink", 0, "webkit text sink"))

static void webkitTextSinkHandleSample(WebKitTextSink* self, GRefPtr<GstSample>&& sample)
{
    auto* priv = self->priv;
    if (!priv->streamId) {
        auto pad = adoptGRef(gst_element_get_static_pad(priv->appSink.get(), "sink"));
        priv->streamId = getStreamIdFromPad(pad.get());
    }

    if (UNLIKELY(!priv->streamId)) {
        GST_WARNING_OBJECT(self, "Unable to handle sample with no stream start event.");
        return;
    }

    // Player private methods that interact with WebCore must run from the main thread. Things can
    // be destroyed before that code runs, including the text sink and priv, so pass everything in a
    // safe way.
    callOnMainThread([mediaPlayerPrivate = ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>(priv->mediaPlayerPrivate), streamId = priv->streamId.value(), sample = WTFMove(sample)]() mutable {
        RefPtr player = mediaPlayerPrivate.get();
        if (!player)
            return;
        player->handleTextSample(WTFMove(sample), streamId);
    });
}

static void webkitTextSinkConstructed(GObject* object)
{
    G_OBJECT_CLASS(webkit_text_sink_parent_class)->constructed(object);

    auto* sink = WEBKIT_TEXT_SINK(object);
    auto* priv = sink->priv;

    priv->appSink = makeGStreamerElement("appsink", nullptr);
    gst_bin_add(GST_BIN_CAST(sink), priv->appSink.get());

    auto pad = adoptGRef(gst_element_get_static_pad(priv->appSink.get(), "sink"));
    gst_element_add_pad(GST_ELEMENT_CAST(sink), gst_ghost_pad_new("sink", pad.get()));

    auto textCaps = adoptGRef(gst_caps_new_empty_simple("application/x-subtitle-vtt"));
    g_object_set(priv->appSink.get(), "emit-signals", TRUE, "enable-last-sample", FALSE, "caps", textCaps.get(), nullptr);

    g_signal_connect(priv->appSink.get(), "new-sample", G_CALLBACK(+[](GstElement* appSink, WebKitTextSink* sink) -> GstFlowReturn {
        webkitTextSinkHandleSample(sink, adoptGRef(gst_app_sink_pull_sample(GST_APP_SINK(appSink))));
        return GST_FLOW_OK;
    }), sink);

    g_signal_connect(priv->appSink.get(), "new-preroll", G_CALLBACK(+[](GstElement* appSink, WebKitTextSink* sink) -> GstFlowReturn {
        webkitTextSinkHandleSample(sink, adoptGRef(gst_app_sink_pull_preroll(GST_APP_SINK(appSink))));
        return GST_FLOW_OK;
    }), sink);

    // We want to get cues as quickly as possible so WebKit has time to handle them,
    // and we don't want cues to block when they come in the wrong order.
    gst_base_sink_set_sync(GST_BASE_SINK_CAST(sink->priv->appSink.get()), false);
}

static gboolean webkitTextSinkQuery(GstElement* element, GstQuery* query)
{
    switch (GST_QUERY_TYPE(query)) {
    case GST_QUERY_DURATION:
    case GST_QUERY_POSITION:
        // Ignore duration and position because we don't want the seek bar to be based on where the cues are.
        return false;
    default:
        return GST_ELEMENT_CLASS(webkit_text_sink_parent_class)->query(element, query);
    }
}

static void webkit_text_sink_class_init(WebKitTextSinkClass* klass)
{
    auto* gobjectClass = G_OBJECT_CLASS(klass);
    auto* elementClass = GST_ELEMENT_CLASS(klass);

    gst_element_class_set_metadata(elementClass, "WebKit text sink", GST_ELEMENT_FACTORY_KLASS_SINK,
        "WebKit's text sink collecting cues encoded in WebVTT by the WebKit text-combiner",
        "Brendan Long <b.long@cablelabs.com>");

    gobjectClass->constructed = GST_DEBUG_FUNCPTR(webkitTextSinkConstructed);
    elementClass->query = GST_DEBUG_FUNCPTR(webkitTextSinkQuery);
}

GstElement* webkitTextSinkNew(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&& player)
{
    auto* element = GST_ELEMENT_CAST(g_object_new(WEBKIT_TYPE_TEXT_SINK, nullptr));
    auto* sink = WEBKIT_TEXT_SINK(element);
    ASSERT(isMainThread());
    sink->priv->mediaPlayerPrivate = WTFMove(player);
    return element;
}

#undef GST_CAT_DEFAULT

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
