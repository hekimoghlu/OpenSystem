/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 9, 2024.
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

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "RTCDtlsTransportBackend.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

ALLOW_UNUSED_PARAMETERS_BEGIN

#include <webrtc/api/scoped_refptr.h>

ALLOW_UNUSED_PARAMETERS_END

namespace webrtc {
class DtlsTransportInterface;
}

namespace WebCore {
class LibWebRTCDtlsTransportBackendObserver;
class LibWebRTCDtlsTransportBackend final : public RTCDtlsTransportBackend, public CanMakeWeakPtr<LibWebRTCDtlsTransportBackend> {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCDtlsTransportBackend);
public:
    explicit LibWebRTCDtlsTransportBackend(rtc::scoped_refptr<webrtc::DtlsTransportInterface>&&);
    ~LibWebRTCDtlsTransportBackend();

private:
    // RTCDtlsTransportBackend
    const void* backend() const final { return m_backend.get(); }
    UniqueRef<RTCIceTransportBackend> iceTransportBackend() final;
    void registerClient(RTCDtlsTransportBackendClient&) final;
    void unregisterClient() final;

    rtc::scoped_refptr<webrtc::DtlsTransportInterface> m_backend;
    RefPtr<LibWebRTCDtlsTransportBackendObserver> m_observer;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
