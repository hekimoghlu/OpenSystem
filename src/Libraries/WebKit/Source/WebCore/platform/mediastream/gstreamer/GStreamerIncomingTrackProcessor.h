/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 25, 2021.
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

#include "GStreamerMediaEndpoint.h"
#include "GStreamerWebRTCCommon.h"
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GStreamerIncomingTrackProcessor : public RefCounted<GStreamerIncomingTrackProcessor> {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerIncomingTrackProcessor);

public:
    static Ref<GStreamerIncomingTrackProcessor> create()
    {
        return adoptRef(*new GStreamerIncomingTrackProcessor());
    }
    ~GStreamerIncomingTrackProcessor() = default;

    void configure(ThreadSafeWeakPtr<GStreamerMediaEndpoint>&&, GRefPtr<GstPad>&&);
    const GRefPtr<GstPad>& pad() const { return m_pad; }

    GstElement* bin() const { return m_bin.get(); }

    const GstStructure* stats();

    bool isDecoding() const { return m_isDecoding; }
    bool isReady() const { return m_isReady; }
    const String& trackId() const { return m_data.trackId; }

private:
    GStreamerIncomingTrackProcessor();

    void retrieveMediaStreamAndTrackIdFromSDP();
    String mediaStreamIdFromPad();

    GRefPtr<GstElement> incomingTrackProcessor();
    GRefPtr<GstElement> createParser();

    void installRtpBufferPadProbe(const GRefPtr<GstPad>&);

    void trackReady();

    ThreadSafeWeakPtr<GStreamerMediaEndpoint> m_endPoint;
    GRefPtr<GstPad> m_pad;
    GRefPtr<GstElement> m_bin;
    WebRTCTrackData m_data;

    std::pair<String, String> m_sdpMsIdAndTrackId;

    bool m_isDecoding { false };
    FloatSize m_videoSize;
    uint64_t m_decodedVideoFrames { 0 };
    GRefPtr<GstElement> m_sink;
    GUniquePtr<GstStructure> m_stats;
    bool m_isReady { false };
};

} // namespace WebCore

#endif // USE(GSTREAMER_WEBRTC)
