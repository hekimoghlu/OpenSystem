/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#include "GStreamerRtpReceiverTransformBackend.h"

#if ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)

#include "NotImplemented.h"

namespace WebCore {

static inline GStreamerRtpReceiverTransformBackend::MediaType mediaTypeFromReceiver(const GstWebRTCRTPReceiver&)
{
    notImplemented();
    return RTCRtpTransformBackend::MediaType::Video;
}

GStreamerRtpReceiverTransformBackend::GStreamerRtpReceiverTransformBackend(const GRefPtr<GstWebRTCRTPReceiver>& rtcReceiver)
    : GStreamerRtpTransformBackend(mediaTypeFromReceiver(*rtcReceiver), Side::Receiver)
    , m_rtcReceiver(rtcReceiver)
{
}

GStreamerRtpReceiverTransformBackend::~GStreamerRtpReceiverTransformBackend()
{
}

void GStreamerRtpReceiverTransformBackend::setTransformableFrameCallback(Callback&& callback)
{
    setInputCallback(WTFMove(callback));
    notImplemented();
}

void GStreamerRtpReceiverTransformBackend::requestKeyFrame()
{
    ASSERT(mediaType() == MediaType::Video);
    notImplemented();
}

} // namespace WebCore

#endif // ENABLE(WEB_RTC) && USE(GSTREAMER_WEBRTC)
