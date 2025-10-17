/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 24, 2023.
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

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "GRefPtrGStreamer.h"

#include <wtf/CompletionHandler.h>
#include <wtf/RefPtr.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>

namespace WebCore {

class RTCStatsReport;

class GStreamerStatsCollector : public ThreadSafeRefCounted<GStreamerStatsCollector, WTF::DestructionThread::Main> {
public:
    using CollectorCallback = CompletionHandler<void(RefPtr<RTCStatsReport>&&)>;
    static Ref<GStreamerStatsCollector> create() { return adoptRef(*new GStreamerStatsCollector()); }

    void setElement(GstElement* element) { m_webrtcBin = element; }
    using PreprocessCallback = Function<GUniquePtr<GstStructure>(const GRefPtr<GstPad>&, const GstStructure*)>;
    void getStats(CollectorCallback&&, const GRefPtr<GstPad>&, PreprocessCallback&&);

private:
    GRefPtr<GstElement> m_webrtcBin;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
