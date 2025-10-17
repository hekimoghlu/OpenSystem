/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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

#include "LibWebRTCRtpTransformBackend.h"
#include <wtf/Ref.h>

ALLOW_UNUSED_PARAMETERS_BEGIN
ALLOW_DEPRECATED_DECLARATIONS_BEGIN

#include <webrtc/api/rtp_receiver_interface.h>

ALLOW_DEPRECATED_DECLARATIONS_END
ALLOW_UNUSED_PARAMETERS_END

namespace webrtc {
class RtpReceiverInterface;
}

namespace WebCore {

class LibWebRTCRtpReceiverTransformBackend final : public LibWebRTCRtpTransformBackend {
public:
    static Ref<LibWebRTCRtpReceiverTransformBackend> create(rtc::scoped_refptr<webrtc::RtpReceiverInterface> receiver) { return adoptRef(*new LibWebRTCRtpReceiverTransformBackend(WTFMove(receiver))); }
    ~LibWebRTCRtpReceiverTransformBackend();

private:
    explicit LibWebRTCRtpReceiverTransformBackend(rtc::scoped_refptr<webrtc::RtpReceiverInterface>);

    // RTCRtpTransformBackend
    void setTransformableFrameCallback(Callback&&) final;
    void requestKeyFrame() final;

    bool m_isRegistered { false };
    rtc::scoped_refptr<webrtc::RtpReceiverInterface> m_rtcReceiver;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
