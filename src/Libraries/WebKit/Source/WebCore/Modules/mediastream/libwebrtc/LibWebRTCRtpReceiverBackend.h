/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include "RTCRtpReceiverBackend.h"
#include <webrtc/api/scoped_refptr.h>
#include <wtf/TZoneMalloc.h>

namespace webrtc {
class RtpReceiverInterface;
}

namespace WebCore {
class Document;
class RealtimeMediaSource;

class LibWebRTCRtpReceiverBackend final : public RTCRtpReceiverBackend {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCRtpReceiverBackend);
public:
    explicit LibWebRTCRtpReceiverBackend(rtc::scoped_refptr<webrtc::RtpReceiverInterface>&&);
    ~LibWebRTCRtpReceiverBackend();

    webrtc::RtpReceiverInterface* rtcReceiver() { return m_rtcReceiver.get(); }

    Ref<RealtimeMediaSource> createSource(Document&);

private:
    RTCRtpParameters getParameters() final;
    Vector<RTCRtpContributingSource> getContributingSources() const final;
    Vector<RTCRtpSynchronizationSource> getSynchronizationSources() const final;
    Ref<RTCRtpTransformBackend> rtcRtpTransformBackend() final;
    std::unique_ptr<RTCDtlsTransportBackend> dtlsTransportBackend() final;

    rtc::scoped_refptr<webrtc::RtpReceiverInterface> m_rtcReceiver;
    RefPtr<RTCRtpTransformBackend> m_transformBackend;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
