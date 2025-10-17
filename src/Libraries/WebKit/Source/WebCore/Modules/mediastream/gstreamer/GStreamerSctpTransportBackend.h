/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#include "RTCSctpTransportBackend.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

typedef struct _GstWebRTCSCTPTransport GstWebRTCSCTPTransport;

namespace WebCore {

class GStreamerSctpTransportBackend final : public RTCSctpTransportBackend, public CanMakeWeakPtr<GStreamerSctpTransportBackend> {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerSctpTransportBackend);
public:
    explicit GStreamerSctpTransportBackend(GRefPtr<GstWebRTCSCTPTransport>&&);
    ~GStreamerSctpTransportBackend();

protected:
    void stateChanged();

private :
    // RTCSctpTransportBackend
    const void* backend() const final { return m_backend.get(); }
    UniqueRef<RTCDtlsTransportBackend> dtlsTransportBackend() final;
    void registerClient(RTCSctpTransportBackendClient&) final;
    void unregisterClient() final;

    GRefPtr<GstWebRTCSCTPTransport> m_backend;
    WeakPtr<RTCSctpTransportBackendClient> m_client;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
