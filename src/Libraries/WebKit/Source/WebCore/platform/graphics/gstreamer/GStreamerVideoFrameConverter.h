/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#pragma once

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "GRefPtrGStreamer.h"
#include <wtf/Forward.h>

namespace WebCore {

class GStreamerVideoFrameConverter {
    friend NeverDestroyed<GStreamerVideoFrameConverter>;

public:
    static GStreamerVideoFrameConverter& singleton();

    WARN_UNUSED_RETURN GRefPtr<GstSample> convert(const GRefPtr<GstSample>&, const GRefPtr<GstCaps>&);

private:
    GStreamerVideoFrameConverter();

    GRefPtr<GstElement> m_pipeline;
    GRefPtr<GstElement> m_src;
    GRefPtr<GstElement> m_sink;
};

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
