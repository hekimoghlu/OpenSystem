/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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
#include "LibWebRTCRtpSenderBackend.h"
#include "RTCRtpTransceiverBackend.h"
#include <wtf/TZoneMalloc.h>

ALLOW_UNUSED_PARAMETERS_BEGIN
ALLOW_DEPRECATED_DECLARATIONS_BEGIN

#include <webrtc/api/rtp_transceiver_interface.h>
#include <webrtc/api/scoped_refptr.h>

ALLOW_DEPRECATED_DECLARATIONS_END
ALLOW_UNUSED_PARAMETERS_END

namespace WebCore {

class LibWebRTCRtpReceiverBackend;

class LibWebRTCRtpTransceiverBackend final : public RTCRtpTransceiverBackend {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCRtpTransceiverBackend);
public:
    explicit LibWebRTCRtpTransceiverBackend(rtc::scoped_refptr<webrtc::RtpTransceiverInterface>&& rtcTransceiver)
        : m_rtcTransceiver(WTFMove(rtcTransceiver))
    {
    }

    std::unique_ptr<LibWebRTCRtpReceiverBackend> createReceiverBackend();
    std::unique_ptr<LibWebRTCRtpSenderBackend> createSenderBackend(LibWebRTCPeerConnectionBackend&, LibWebRTCRtpSenderBackend::Source&&);

    webrtc::RtpTransceiverInterface* rtcTransceiver() { return m_rtcTransceiver.get(); }

private:
    RTCRtpTransceiverDirection direction() const final;
    std::optional<RTCRtpTransceiverDirection> currentDirection() const final;
    void setDirection(RTCRtpTransceiverDirection) final;
    String mid() final;
    void stop() final;
    bool stopped() const final;
    ExceptionOr<void> setCodecPreferences(const Vector<RTCRtpCodecCapability>&) final;

    rtc::scoped_refptr<webrtc::RtpTransceiverInterface> m_rtcTransceiver;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
