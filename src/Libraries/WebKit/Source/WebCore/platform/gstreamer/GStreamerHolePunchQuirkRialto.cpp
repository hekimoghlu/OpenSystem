/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 29, 2025.
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
#include "GStreamerHolePunchQuirkRialto.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include "MediaPlayerPrivateGStreamer.h"
#include <wtf/text/MakeString.h>

namespace WebCore {

GstElement* GStreamerHolePunchQuirkRialto::createHolePunchVideoSink(bool isLegacyPlaybin, const MediaPlayer* player)
{
    AtomString value;
    bool isPIPRequested = player && player->doesHaveAttribute("pip"_s, &value) && equalLettersIgnoringASCIICase(value, "true"_s);
    if (isLegacyPlaybin && !isPIPRequested)
        return nullptr;

    // Rialto using holepunch.
    GstElement* videoSink = makeGStreamerElement("rialtomsevideosink", nullptr);
    if (isPIPRequested)
        g_object_set(G_OBJECT(videoSink), "maxVideoWidth", 640, "maxVideoHeight", 480, "has-drm", FALSE, nullptr);
    return videoSink;
}

bool GStreamerHolePunchQuirkRialto::setHolePunchVideoRectangle(GstElement* videoSink, const IntRect& rect)
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
