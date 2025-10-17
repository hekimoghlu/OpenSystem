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
#pragma once

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCMacros.h"
#include "RTCIceTransportBackend.h"
#include <wtf/TZoneMalloc.h>

ALLOW_UNUSED_PARAMETERS_BEGIN

#include <webrtc/api/scoped_refptr.h>

ALLOW_UNUSED_PARAMETERS_END

namespace webrtc {
class IceTransportInterface;
}

namespace WebCore {

class LibWebRTCIceTransportBackendObserver;

class LibWebRTCIceTransportBackend final : public RTCIceTransportBackend {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCIceTransportBackend);
public:
    explicit LibWebRTCIceTransportBackend(rtc::scoped_refptr<webrtc::IceTransportInterface>&&);
    ~LibWebRTCIceTransportBackend();

private:
    // RTCIceTransportBackend
    const void* backend() const final { return m_backend.get(); }
    void registerClient(RTCIceTransportBackendClient&) final;
    void unregisterClient() final;

    rtc::scoped_refptr<webrtc::IceTransportInterface> m_backend;
    RefPtr<LibWebRTCIceTransportBackendObserver> m_observer;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
