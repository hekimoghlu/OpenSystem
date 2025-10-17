/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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

#include "RTCRtpTransformBackend.h"
#include <wtf/Lock.h>

namespace WebCore {

class GStreamerRtpTransformBackend : public RTCRtpTransformBackend {
protected:
    GStreamerRtpTransformBackend(MediaType, Side);
    void setInputCallback(Callback&&);

protected:
    MediaType mediaType() const final { return m_mediaType; }

private:
    // RTCRtpTransformBackend
    void processTransformedFrame(RTCRtpTransformableFrame&) final;
    void clearTransformableFrameCallback() final;
    Side side() const final { return m_side; }

    MediaType m_mediaType;
    Side m_side;

    Lock m_inputCallbackLock;
    Callback m_inputCallback;

    Lock m_outputCallbackLock;
};

inline GStreamerRtpTransformBackend::GStreamerRtpTransformBackend(MediaType mediaType, Side side)
    : m_mediaType(mediaType)
    , m_side(side)
{
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
