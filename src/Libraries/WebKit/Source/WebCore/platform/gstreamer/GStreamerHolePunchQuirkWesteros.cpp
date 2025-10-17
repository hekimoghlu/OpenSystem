/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 30, 2023.
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
#include "GStreamerHolePunchQuirkWesteros.h"
#include "MediaPlayerPrivateGStreamer.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

GstElement* GStreamerHolePunchQuirkWesteros::createHolePunchVideoSink(bool isLegacyPlaybin, const MediaPlayer* player)
{
    AtomString val;
    bool isPIPRequested = player && player->doesHaveAttribute("pip"_s, &val) && equalLettersIgnoringASCIICase(val, "true"_s);
    if (isLegacyPlaybin && !isPIPRequested)
        return nullptr;

    // Westeros using holepunch.
    GstElement* videoSink = makeGStreamerElement("westerossink", "WesterosVideoSink");
    g_object_set(videoSink, "zorder", 0.0f, nullptr);
    if (isPIPRequested) {
        g_object_set(videoSink, "res-usage", 0u, nullptr);
        // Set context for pipelines that use ERM in decoder elements.
        auto context = adoptGRef(gst_context_new("erm", FALSE));
        auto contextStructure = gst_context_writable_structure(context.get());
        gst_structure_set(contextStructure, "res-usage", G_TYPE_UINT, 0x0u, nullptr);
        auto playerPrivate = reinterpret_cast<const MediaPlayerPrivateGStreamer*>(player->playerPrivate());
        gst_element_set_context(playerPrivate->pipeline(), context.get());
    }
    return videoSink;
}

bool GStreamerHolePunchQuirkWesteros::setHolePunchVideoRectangle(GstElement* videoSink, const IntRect& rect)
{
    if (UNLIKELY(!gstObjectHasProperty(videoSink, "rectangle")))
        return false;

    auto rectString = makeString(rect.x(), ',', rect.y(), ',', rect.width(), ',', rect.height());
    g_object_set(videoSink, "rectangle", rectString.ascii().data(), nullptr);
    return true;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
