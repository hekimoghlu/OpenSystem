/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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
#include "GStreamerQuirkWesteros.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <wtf/OptionSet.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_westeros_quirks_debug);
#define GST_CAT_DEFAULT webkit_westeros_quirks_debug

GStreamerQuirkWesteros::GStreamerQuirkWesteros()
{
    GST_DEBUG_CATEGORY_INIT(webkit_westeros_quirks_debug, "webkitquirkswesteros", 0, "WebKit Westeros Quirks");

    auto westerosFactory = adoptGRef(gst_element_factory_find("westerossink"));
    if (UNLIKELY(!westerosFactory))
        return;

    gst_object_unref(gst_plugin_feature_load(GST_PLUGIN_FEATURE(westerosFactory.get())));
    for (auto* t = gst_element_factory_get_static_pad_templates(westerosFactory.get()); t; t = g_list_next(t)) {
        auto* padtemplate = static_cast<GstStaticPadTemplate*>(t->data);
        if (padtemplate->direction != GST_PAD_SINK)
            continue;
        if (m_sinkCaps)
            m_sinkCaps = adoptGRef(gst_caps_merge(m_sinkCaps.leakRef(), gst_static_caps_get(&padtemplate->static_caps)));
        else
            m_sinkCaps = adoptGRef(gst_static_caps_get(&padtemplate->static_caps));
    }
}

void GStreamerQuirkWesteros::configureElement(GstElement* element, const OptionSet<ElementRuntimeCharacteristics>& characteristics)
{
    auto view = StringView::fromLatin1(GST_ELEMENT_NAME(element));
    if (view.startsWith("uridecodebin3"_s)) {
        GRefPtr<GstCaps> defaultCaps;
        g_object_get(element, "caps", &defaultCaps.outPtr(), nullptr);
        defaultCaps = adoptGRef(gst_caps_merge(gst_caps_ref(m_sinkCaps.get()), defaultCaps.leakRef()));
        GST_INFO("Setting stop caps to %" GST_PTR_FORMAT, defaultCaps.get());
        g_object_set(element, "caps", defaultCaps.get(), nullptr);
        return;
    }

    if (!characteristics.contains(ElementRuntimeCharacteristics::IsMediaStream))
        return;

    if (!g_strcmp0(G_OBJECT_TYPE_NAME(G_OBJECT(element)), "GstWesterosSink") && gstObjectHasProperty(element, "immediate-output")) {
        GST_INFO("Enable 'immediate-output' in WesterosSink");
        g_object_set(element, "immediate-output", TRUE, nullptr);
    }
}

std::optional<bool> GStreamerQuirkWesteros::isHardwareAccelerated(GstElementFactory* factory)
{
    auto view = StringView::fromLatin1(GST_OBJECT_NAME(factory));
    if (view.startsWith("westeros"_s))
        return true;

    return std::nullopt;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
