/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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

#include "LibWebRTCProvider.h"

#if USE(LIBWEBRTC)

#include <wtf/TZoneMalloc.h>

namespace webrtc {
class VideoDecoderFactory;
class VideoEncoderFactory;
}

namespace WebCore {

class WEBCORE_EXPORT LibWebRTCProviderCocoa : public LibWebRTCProvider {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(LibWebRTCProviderCocoa, WEBCORE_EXPORT);
public:
    LibWebRTCProviderCocoa() = default;
    ~LibWebRTCProviderCocoa();

    std::unique_ptr<webrtc::VideoDecoderFactory> createDecoderFactory() override;
    std::unique_ptr<webrtc::VideoEncoderFactory> createEncoderFactory() override;

private:
    std::optional<MediaCapabilitiesInfo> computeVPParameters(const VideoConfiguration&) final;
    bool isVPSoftwareDecoderSmooth(const VideoConfiguration&) final;
};

} // namespace WebCore

#endif
