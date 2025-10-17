/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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

#include "EventTarget.h"
#include "SecurityOrigin.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {
class RTCController;
}

namespace WebCore {

class Document;
class RTCPeerConnection;
class WeakPtrImplWithEventTargetData;

#if USE(LIBWEBRTC)
class LibWebRTCLogSink;
#endif

#if USE(GSTREAMER_WEBRTC)
class GStreamerWebRTCLogSink;
#endif

class RTCController : public RefCountedAndCanMakeWeakPtr<RTCController> {
public:
    static Ref<RTCController> create() { return adoptRef(*new RTCController); }

#if ENABLE(WEB_RTC)
    ~RTCController();

    void reset(bool shouldFilterICECandidates);

    void add(RTCPeerConnection&);
    void remove(RTCPeerConnection&);

    WEBCORE_EXPORT void disableICECandidateFilteringForAllOrigins();
    WEBCORE_EXPORT void disableICECandidateFilteringForDocument(Document&);
    WEBCORE_EXPORT void enableICECandidateFiltering();

    using LogCallback = Function<void(String&& logType, String&& logMessage, String&& logLevel, RefPtr<RTCPeerConnection>&&)>;
    void startGatheringLogs(Document&, LogCallback&&);
    void stopGatheringLogs();
#endif

private:
    RTCController();

#if ENABLE(WEB_RTC)
    void startGatheringStatLogs(RTCPeerConnection&);
    bool shouldDisableICECandidateFiltering(Document&);

    void stopLoggingWebRTCLogs();

    struct PeerConnectionOrigin {
        Ref<SecurityOrigin> topOrigin;
        Ref<SecurityOrigin> clientOrigin;
    };
    Vector<PeerConnectionOrigin> m_filteringDisabledOrigins;
    WeakHashSet<RTCPeerConnection, WeakPtrImplWithEventTargetData> m_peerConnections;
    bool m_shouldFilterICECandidates { true };

    LogCallback m_callback;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_gatheringLogsDocument;
#if USE(LIBWEBRTC)
    std::unique_ptr<LibWebRTCLogSink> m_logSink;
#endif
#if USE(GSTREAMER_WEBRTC)
    std::unique_ptr<GStreamerWebRTCLogSink> m_logSink;
#endif
#endif // ENABLE(WEB_RTC)
};

} // namespace WebCore
