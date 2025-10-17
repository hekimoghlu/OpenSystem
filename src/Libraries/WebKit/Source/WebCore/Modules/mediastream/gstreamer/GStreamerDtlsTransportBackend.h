/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#include "GStreamerPeerConnectionBackend.h"
#include "RTCDtlsTransportBackend.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class GStreamerDtlsTransportBackend;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::GStreamerDtlsTransportBackend> : std::true_type { };
}

namespace WebCore {

class GStreamerDtlsTransportBackendObserver;

class GStreamerDtlsTransportBackend final : public RTCDtlsTransportBackend, public CanMakeWeakPtr<GStreamerDtlsTransportBackend> {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerDtlsTransportBackend);

public:
    explicit GStreamerDtlsTransportBackend(GRefPtr<GstWebRTCDTLSTransport>&&);
    ~GStreamerDtlsTransportBackend();

private:
    // RTCDtlsTransportBackend
    const void* backend() const final { return m_backend.get(); }
    UniqueRef<RTCIceTransportBackend> iceTransportBackend() final;
    void registerClient(RTCDtlsTransportBackendClient&) final;
    void unregisterClient() final;

    GRefPtr<GstWebRTCDTLSTransport> m_backend;
    RefPtr<GStreamerDtlsTransportBackendObserver> m_observer;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
