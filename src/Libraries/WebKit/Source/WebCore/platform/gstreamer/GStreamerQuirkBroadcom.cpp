/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "GStreamerQuirkBroadcom.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <wtf/OptionSet.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_broadcom_quirks_debug);
#define GST_CAT_DEFAULT webkit_broadcom_quirks_debug

GStreamerQuirkBroadcom::GStreamerQuirkBroadcom()
{
    GST_DEBUG_CATEGORY_INIT(webkit_broadcom_quirks_debug, "webkitquirksbroadcom", 0, "WebKit Broadcom Quirks");
    m_disallowedWebAudioDecoders = { "brcmaudfilter"_s };
}

void GStreamerQuirkBroadcom::configureElement(GstElement* element, const OptionSet<ElementRuntimeCharacteristics>& characteristics)
{
    auto view = StringView::fromLatin1(GST_ELEMENT_NAME(element));
    if (!g_strcmp0(G_OBJECT_TYPE_NAME(element), "Gstbrcmaudiosink"))
        g_object_set(G_OBJECT(element), "async", TRUE, nullptr);
    else if (view.startsWith("brcmaudiodecoder"_s)) {
        // Limit BCM audio decoder buffering to 1sec so live progressive playback can start faster.
        if (characteristics.contains(ElementRuntimeCharacteristics::IsLiveStream))
            g_object_set(G_OBJECT(element), "limit_buffering_ms", 1000, nullptr);
    }

    if (!characteristics.contains(ElementRuntimeCharacteristics::IsMediaStream))
        return;

    if (!g_strcmp0(G_OBJECT_TYPE_NAME(G_OBJECT(element)), "GstBrcmPCMSink") && gstObjectHasProperty(element, "low_latency")) {
        GST_DEBUG("Set 'low_latency' in brcmpcmsink");
        g_object_set(element, "low_latency", TRUE, "low_latency_max_queued_ms", 60, nullptr);
    }
}

std::optional<bool> GStreamerQuirkBroadcom::isHardwareAccelerated(GstElementFactory* factory)
{
    auto view = StringView::fromLatin1(GST_OBJECT_NAME(factory));
    if (view.startsWith("brcm"_s))
        return true;

    return std::nullopt;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
