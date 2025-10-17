/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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

#include "WebRTCProvider.h"
#include <wtf/TZoneMalloc.h>

#if USE(GSTREAMER_WEBRTC)

namespace WebCore {

class WEBCORE_EXPORT GStreamerWebRTCProvider : public WebRTCProvider {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerWebRTCProvider);
public:
    std::optional<RTCRtpCapabilities> receiverCapabilities(const String& kind) final;
    std::optional<RTCRtpCapabilities> senderCapabilities(const String& kind) final;

private:
    void initializeAudioDecodingCapabilities() final;
    void initializeVideoDecodingCapabilities() final;
    void initializeAudioEncodingCapabilities() final;
    void initializeVideoEncodingCapabilities() final;
    std::optional<MediaCapabilitiesDecodingInfo> videoDecodingCapabilitiesOverride(const VideoConfiguration&) final;
};

} // namespace WebCore

#endif
