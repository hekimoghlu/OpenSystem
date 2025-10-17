/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#include "GStreamerQuirkRialto.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "WebKitAudioSinkGStreamer.h"
#include <wtf/OptionSet.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_rialto_quirks_debug);
#define GST_CAT_DEFAULT webkit_rialto_quirks_debug

GStreamerQuirkRialto::GStreamerQuirkRialto()
{
    GST_DEBUG_CATEGORY_INIT(webkit_rialto_quirks_debug, "webkitquirksrialto", 0, "WebKit Rialto Quirks");

    std::array<const char *, 2> rialtoSinks = { "rialtomsevideosink", "rialtomseaudiosink" };

    for (const auto* sink : rialtoSinks) {
        auto sinkFactory = adoptGRef(gst_element_factory_find(sink));
        if (UNLIKELY(!sinkFactory))
            continue;

        gst_object_unref(gst_plugin_feature_load(GST_PLUGIN_FEATURE(sinkFactory.get())));
        for (auto* padTemplateListElement = gst_element_factory_get_static_pad_templates(sinkFactory.get());
            padTemplateListElement; padTemplateListElement = g_list_next(padTemplateListElement)) {

            auto* padTemplate = static_cast<GstStaticPadTemplate*>(padTemplateListElement->data);
            if (padTemplate->direction != GST_PAD_SINK)
                continue;
            GRefPtr<GstCaps> templateCaps = adoptGRef(gst_static_caps_get(&padTemplate->static_caps));
            if (!templateCaps)
                continue;
            if (gst_caps_is_empty(templateCaps.get()) || gst_caps_is_any(templateCaps.get()))
                continue;
            if (m_sinkCaps)
                m_sinkCaps = adoptGRef(gst_caps_merge(m_sinkCaps.leakRef(), templateCaps.leakRef()));
            else
                m_sinkCaps = WTFMove(templateCaps);
        }
    }
}

void GStreamerQuirkRialto::configureElement(GstElement* element, const OptionSet<ElementRuntimeCharacteristics>&)
{
    if (!g_strcmp0(G_OBJECT_TYPE_NAME(G_OBJECT(element)), "GstURIDecodeBin3")) {
        GRefPtr<GstCaps> defaultCaps;
        g_object_get(element, "caps", &defaultCaps.outPtr(), nullptr);
        defaultCaps = adoptGRef(gst_caps_merge(gst_caps_ref(m_sinkCaps.get()), defaultCaps.leakRef()));
        GST_INFO("Setting stop caps to %" GST_PTR_FORMAT, defaultCaps.get());
        g_object_set(element, "caps", defaultCaps.get(), nullptr);
    }
}

GstElement* GStreamerQuirkRialto::createAudioSink()
{
    auto sink = makeGStreamerElement("rialtomseaudiosink", nullptr);
    RELEASE_ASSERT_WITH_MESSAGE(sink, "rialtomseaudiosink should be available in the system but it is not");
    return sink;
}

GstElement* GStreamerQuirkRialto::createWebAudioSink()
{
    if (GstElement* sink = webkitAudioSinkNew())
        return sink;

    auto sink = makeGStreamerElement("rialtowebaudiosink", nullptr);
    RELEASE_ASSERT_WITH_MESSAGE(sink, "rialtowebaudiosink should be available in the system but it is not");
    return sink;
}

std::optional<bool> GStreamerQuirkRialto::isHardwareAccelerated(GstElementFactory* factory)
{
    auto view = StringView::fromLatin1(GST_OBJECT_NAME(factory));
    if (view.startsWith("rialto"_s))
        return true;

    return std::nullopt;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
