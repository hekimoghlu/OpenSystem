/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 23, 2023.
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
#include "RTCRtpTransformBackend.h"
#include <webrtc/api/scoped_refptr.h>
#include <wtf/Lock.h>
#include <wtf/StdUnorderedMap.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#include <webrtc/api/frame_transformer_interface.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

namespace WebCore {

class LibWebRTCRtpTransformBackend : public RTCRtpTransformBackend, public webrtc::FrameTransformerInterface {
protected:
    LibWebRTCRtpTransformBackend(MediaType, Side);
    void setInputCallback(Callback&&);

protected:
    MediaType mediaType() const final { return m_mediaType; }

private:
    void sendFrameToOutput(std::unique_ptr<webrtc::TransformableFrameInterface>&&);
    void addOutputCallback(rtc::scoped_refptr<webrtc::TransformedFrameCallback>&&, uint32_t ssrc);
    void removeOutputCallback(uint32_t ssrc);

    // RTCRtpTransformBackend
    void processTransformedFrame(RTCRtpTransformableFrame&) final;
    void clearTransformableFrameCallback() final;
    Side side() const final { return m_side; }

    // webrtc::FrameTransformerInterface
    void Transform(std::unique_ptr<webrtc::TransformableFrameInterface>) final;
    void RegisterTransformedFrameCallback(rtc::scoped_refptr<webrtc::TransformedFrameCallback>) final;
    void RegisterTransformedFrameSinkCallback(rtc::scoped_refptr<webrtc::TransformedFrameCallback>, uint32_t ssrc) final;
    void UnregisterTransformedFrameCallback() final;
    void UnregisterTransformedFrameSinkCallback(uint32_t ssrc) final;
    void AddRef() const final { ref(); }
    webrtc::RefCountReleaseStatus Release() const final;

    MediaType m_mediaType;
    Side m_side;

    Lock m_inputCallbackLock;
    Callback m_inputCallback WTF_GUARDED_BY_LOCK(m_inputCallbackLock);

    Lock m_outputCallbacksLock;
    StdUnorderedMap<uint32_t, rtc::scoped_refptr<webrtc::TransformedFrameCallback>> m_outputCallbacks WTF_GUARDED_BY_LOCK(m_outputCallbacksLock);
};

inline LibWebRTCRtpTransformBackend::LibWebRTCRtpTransformBackend(MediaType mediaType, Side side)
    : m_mediaType(mediaType)
    , m_side(side)
{
}

inline webrtc::RefCountReleaseStatus LibWebRTCRtpTransformBackend::Release() const
{
    deref();
    return webrtc::RefCountReleaseStatus::kOtherRefsRemained;
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
