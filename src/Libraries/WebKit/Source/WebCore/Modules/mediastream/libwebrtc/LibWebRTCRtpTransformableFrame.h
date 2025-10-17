/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 9, 2025.
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

#include "RTCRtpTransformableFrame.h"
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>

namespace webrtc {
class TransformableFrameInterface;
}

namespace WebCore {

class LibWebRTCRtpTransformableFrame final : public RTCRtpTransformableFrame {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCRtpTransformableFrame);
public:
    static Ref<LibWebRTCRtpTransformableFrame> create(std::unique_ptr<webrtc::TransformableFrameInterface>&& frame, bool isAudioSenderFrame) { return adoptRef(*new LibWebRTCRtpTransformableFrame(WTFMove(frame), isAudioSenderFrame)); }
    ~LibWebRTCRtpTransformableFrame();

    std::unique_ptr<webrtc::TransformableFrameInterface> takeRTCFrame();

private:
    LibWebRTCRtpTransformableFrame(std::unique_ptr<webrtc::TransformableFrameInterface>&&, bool isAudioSenderFrame);

    // RTCRtpTransformableFrame
    std::span<const uint8_t> data() const final;
    void setData(std::span<const uint8_t>) final;
    bool isKeyFrame() const final;
    uint64_t timestamp() const final;
    RTCEncodedAudioFrameMetadata audioMetadata() const final;
    RTCEncodedVideoFrameMetadata videoMetadata() const final;

    std::unique_ptr<webrtc::TransformableFrameInterface> m_rtcFrame;
    bool m_isAudioSenderFrame;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(LIBWEBRTC)
