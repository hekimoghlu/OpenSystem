/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "LibWebRTCRtpTransformBackend.h"

#if ENABLE(WEB_RTC) && USE(LIBWEBRTC)

#include "LibWebRTCRtpTransformableFrame.h"

namespace WebCore {

void LibWebRTCRtpTransformBackend::setInputCallback(Callback&& callback)
{
    Locker locker { m_inputCallbackLock };
    m_inputCallback = WTFMove(callback);
}

void LibWebRTCRtpTransformBackend::clearTransformableFrameCallback()
{
    setInputCallback({ });
}

void LibWebRTCRtpTransformBackend::addOutputCallback(rtc::scoped_refptr<webrtc::TransformedFrameCallback>&& callback, uint32_t ssrc)
{
    Locker locker { m_outputCallbacksLock };
    m_outputCallbacks.insert_or_assign(ssrc, WTFMove(callback));
}

void LibWebRTCRtpTransformBackend::removeOutputCallback(uint32_t ssrc)
{
    Locker locker { m_outputCallbacksLock };
    m_outputCallbacks.erase(ssrc);
}

void LibWebRTCRtpTransformBackend::sendFrameToOutput(std::unique_ptr<webrtc::TransformableFrameInterface>&& rtcFrame)
{
    Locker locker { m_outputCallbacksLock };
    if (m_outputCallbacks.size() == 1) {
        m_outputCallbacks.begin()->second->OnTransformedFrame(WTFMove(rtcFrame));
        return;
    }
    auto iterator = m_outputCallbacks.find(rtcFrame->GetSsrc());
    if (iterator != m_outputCallbacks.end())
        iterator->second->OnTransformedFrame(WTFMove(rtcFrame));
}

void LibWebRTCRtpTransformBackend::processTransformedFrame(RTCRtpTransformableFrame& frame)
{
    if (auto rtcFrame = static_cast<LibWebRTCRtpTransformableFrame&>(frame).takeRTCFrame())
        sendFrameToOutput(WTFMove(rtcFrame));
}

void LibWebRTCRtpTransformBackend::Transform(std::unique_ptr<webrtc::TransformableFrameInterface> rtcFrame)
{
    {
        Locker locker { m_inputCallbackLock };
        if (m_inputCallback) {
            m_inputCallback(LibWebRTCRtpTransformableFrame::create(WTFMove(rtcFrame), m_mediaType == MediaType::Audio && m_side == Side::Sender));
            return;
        }
    }
    // In case of no input callback, make the transform a no-op.
    sendFrameToOutput(WTFMove(rtcFrame));
}

void LibWebRTCRtpTransformBackend::RegisterTransformedFrameCallback(rtc::scoped_refptr<webrtc::TransformedFrameCallback> callback)
{
    addOutputCallback(WTFMove(callback), 0);
}

void LibWebRTCRtpTransformBackend::RegisterTransformedFrameSinkCallback(rtc::scoped_refptr<webrtc::TransformedFrameCallback> callback, uint32_t ssrc)
{
    addOutputCallback(WTFMove(callback), ssrc);
}

void LibWebRTCRtpTransformBackend::UnregisterTransformedFrameCallback()
{
    removeOutputCallback(0);
}

void LibWebRTCRtpTransformBackend::UnregisterTransformedFrameSinkCallback(uint32_t ssrc)
{
    removeOutputCallback(ssrc);
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
