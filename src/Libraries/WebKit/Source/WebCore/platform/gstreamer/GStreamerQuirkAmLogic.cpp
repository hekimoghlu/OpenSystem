/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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
#include "GStreamerQuirkAmLogic.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <wtf/OptionSet.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_amlogic_quirks_debug);
#define GST_CAT_DEFAULT webkit_amlogic_quirks_debug

GStreamerQuirkAmLogic::GStreamerQuirkAmLogic()
{
    GST_DEBUG_CATEGORY_INIT(webkit_amlogic_quirks_debug, "webkitquirksamlogic", 0, "WebKit AmLogic Quirks");
}

GstElement* GStreamerQuirkAmLogic::createWebAudioSink()
{
    // autoaudiosink changes child element state to READY internally in auto detection phase
    // that causes resource acquisition in some cases interrupting any playback already running.
    // On Amlogic we need to set direct-mode=false prop before changing state to READY
    // but this is not possible with autoaudiosink.
    auto sink = makeGStreamerElement("amlhalasink", nullptr);
    RELEASE_ASSERT_WITH_MESSAGE(sink, "amlhalasink should be available in the system but it is not");
    g_object_set(sink, "direct-mode", FALSE, nullptr);
    return sink;
}

void GStreamerQuirkAmLogic::configureElement(GstElement* element, const OptionSet<ElementRuntimeCharacteristics>& characteristics)
{
    if (gstObjectHasProperty(element, "disable-xrun")) {
        GST_INFO("Set property disable-xrun to TRUE");
        g_object_set(element, "disable-xrun", TRUE, nullptr);
    }

    if (characteristics.contains(ElementRuntimeCharacteristics::HasVideo) && gstObjectHasProperty(element, "wait-video")) {
        GST_INFO("Set property wait-video to TRUE");
        g_object_set(element, "wait-video", TRUE, nullptr);
    }
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
