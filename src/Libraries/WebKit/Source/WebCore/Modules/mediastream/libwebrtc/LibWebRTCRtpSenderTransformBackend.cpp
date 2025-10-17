/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "config.h"
#include "LibWebRTCRtpSenderTransformBackend.h"
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCRtpTransformableFrame.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(LibWebRTCRtpSenderTransformBackend);

static inline LibWebRTCRtpSenderTransformBackend::MediaType mediaTypeFromSender(const webrtc::RtpSenderInterface& sender)
{
    return sender.media_type() == cricket::MEDIA_TYPE_AUDIO ? RTCRtpTransformBackend::MediaType::Audio : RTCRtpTransformBackend::MediaType::Video;
}

LibWebRTCRtpSenderTransformBackend::LibWebRTCRtpSenderTransformBackend(rtc::scoped_refptr<webrtc::RtpSenderInterface> rtcSender)
    : LibWebRTCRtpTransformBackend(mediaTypeFromSender(*rtcSender), Side::Sender)
    , m_rtcSender(WTFMove(rtcSender))
{
}

LibWebRTCRtpSenderTransformBackend::~LibWebRTCRtpSenderTransformBackend()
{
}

void LibWebRTCRtpSenderTransformBackend::setTransformableFrameCallback(Callback&& callback)
{
    setInputCallback(WTFMove(callback));
    if (m_isRegistered)
        return;

    m_isRegistered = true;
    m_rtcSender->SetEncoderToPacketizerFrameTransformer(rtc::scoped_refptr<webrtc::FrameTransformerInterface>(this));
}

void LibWebRTCRtpSenderTransformBackend::requestKeyFrame()
{
    ASSERT(mediaType() == MediaType::Video);
    m_rtcSender->GenerateKeyFrame({ });
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
