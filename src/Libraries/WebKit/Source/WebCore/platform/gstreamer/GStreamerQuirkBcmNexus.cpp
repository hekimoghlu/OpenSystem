/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 2, 2025.
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
#include "GStreamerQuirkBcmNexus.h"

#if USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <wtf/OptionSet.h>

namespace WebCore {

GST_DEBUG_CATEGORY_STATIC(webkit_bcmnexus_quirks_debug);
#define GST_CAT_DEFAULT webkit_bcmnexus_quirks_debug

GStreamerQuirkBcmNexus::GStreamerQuirkBcmNexus()
{
    GST_DEBUG_CATEGORY_INIT(webkit_bcmnexus_quirks_debug, "webkitquirksbcmnexus", 0, "WebKit BcmNexus Quirks");
    m_disallowedWebAudioDecoders = { "brcmaudfilter"_s };

    auto registry = gst_registry_get();
    auto brcmaudfilter = adoptGRef(gst_registry_lookup_feature(registry, "brcmaudfilter"));
    auto mpegaudioparse = adoptGRef(gst_registry_lookup_feature(registry, "mpegaudioparse"));

    if (brcmaudfilter && mpegaudioparse) {
        GST_INFO("Overriding mpegaudioparse rank with brcmaudfilter rank + 1");
        gst_plugin_feature_set_rank(mpegaudioparse.get(), gst_plugin_feature_get_rank(brcmaudfilter.get()) + 1);
    }
}

std::optional<bool> GStreamerQuirkBcmNexus::isHardwareAccelerated(GstElementFactory* factory)
{
    auto view = StringView::fromLatin1(GST_OBJECT_NAME(factory));
    if (view.startsWith("brcm"_s))
        return true;

    return std::nullopt;
}

#undef GST_CAT_DEFAULT

} // namespace WebCore

#endif // USE(GSTREAMER)
