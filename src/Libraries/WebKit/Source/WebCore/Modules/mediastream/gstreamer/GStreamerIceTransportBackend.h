/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include "RTCIceTransportBackend.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class GStreamerIceTransportBackend;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::GStreamerIceTransportBackend> : std::true_type { };
}

namespace WebCore {

class GStreamerIceTransportBackend final : public RTCIceTransportBackend, public CanMakeWeakPtr<GStreamerIceTransportBackend> {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerIceTransportBackend);

public:
    explicit GStreamerIceTransportBackend(GRefPtr<GstWebRTCDTLSTransport>&&);
    ~GStreamerIceTransportBackend();

private:
    // RTCIceTransportBackend
    const void* backend() const final { return m_backend.get(); }

    void registerClient(RTCIceTransportBackendClient&) final;
    void unregisterClient() final;

    void stateChanged() const;
    void gatheringStateChanged() const;
    void iceTransportChanged();
    void selectedCandidatePairChanged();

    GRefPtr<GstWebRTCDTLSTransport> m_backend;
    GRefPtr<GstWebRTCICETransport> m_iceTransport;
    WeakPtr<RTCIceTransportBackendClient> m_client;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
