/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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

#if USE(GSTREAMER_WEBRTC)

#include <gst/gstinfo.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GStreamerWebRTCLogSink {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerWebRTCLogSink);

public:
    using LogCallback = Function<void(const String&, const String&)>;
    explicit GStreamerWebRTCLogSink(LogCallback&&);

    ~GStreamerWebRTCLogSink();

    void start();
    void stop();

private:
    LogCallback m_callback;
    bool m_isGstDebugActive { false };
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
