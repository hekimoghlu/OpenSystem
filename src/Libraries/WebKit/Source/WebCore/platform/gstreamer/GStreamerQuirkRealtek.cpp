/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#include "GStreamerQuirkRealtek.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <wtf/OptionSet.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_realtek_quirks_debug);
#define GST_CAT_DEFAULT webkit_realtek_quirks_debug

GStreamerQuirkRealtek::GStreamerQuirkRealtek()
{
    GST_DEBUG_CATEGORY_INIT(webkit_realtek_quirks_debug, "webkitquirksrealtek", 0, "WebKit Realtek Quirks");
    m_disallowedWebAudioDecoders = {
        "omxaacdec"_s,
        "omxac3dec"_s,
        "omxac4dec"_s,
        "omxeac3dec"_s,
        "omxflacdec"_s,
        "omxlpcmdec"_s,
        "omxmp3dec"_s,
        "omxopusdec"_s,
        "omxvorbisdec"_s,
    };
}

GstElement* GStreamerQuirkRealtek::createWebAudioSink()
{
    auto sink = makeGStreamerElement("rtkaudiosink", nullptr);
    RELEASE_ASSERT_WITH_MESSAGE(sink, "rtkaudiosink should be available in the system but it is not");
    g_object_set(sink, "media-tunnel", FALSE, "audio-service", TRUE, nullptr);
    return sink;
}

void GStreamerQuirkRealtek::configureElement(GstElement* element, const OptionSet<ElementRuntimeCharacteristics>& characteristics)
{
    if (!characteristics.contains(ElementRuntimeCharacteristics::IsMediaStream))
        return;

    if (gstObjectHasProperty(element, "media-tunnel")) {
        GST_INFO("Enable 'immediate-output' in rtkaudiosink");
        g_object_set(element, "media-tunnel", FALSE, "audio-service", TRUE, "lowdelay-sync-mode", TRUE, nullptr);
    }

    if (gstObjectHasProperty(element, "lowdelay-mode")) {
        GST_INFO("Enable 'lowdelay-mode' in rtk omx decoder");
        g_object_set(element, "lowdelay-mode", TRUE, nullptr);
    }
}

std::optional<bool> GStreamerQuirkRealtek::isHardwareAccelerated(GstElementFactory* factory)
{
    auto view = StringView::fromLatin1(GST_OBJECT_NAME(factory));
    if (view.startsWith("omx"_s))
        return true;

    return std::nullopt;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
