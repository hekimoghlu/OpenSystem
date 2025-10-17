/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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

#if USE(LIBWEBRTC) && USE(GSTREAMER)

#include "LibWebRTCMacros.h"
#include "api/video_codecs/video_decoder_factory.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GStreamerVideoDecoderFactory : public webrtc::VideoDecoderFactory {
    WTF_MAKE_TZONE_ALLOCATED(GStreamerVideoDecoderFactory);
public:
    GStreamerVideoDecoderFactory(bool isSupportingVP9Profile0, bool isSupportingVP9Profile2);

private:
    std::unique_ptr<webrtc::VideoDecoder> Create(const webrtc::Environment&, const webrtc::SdpVideoFormat&) final;
    std::vector<webrtc::SdpVideoFormat> GetSupportedFormats() const final;

    bool m_isSupportingVP9Profile0;
    bool m_isSupportingVP9Profile2;
};
}

#endif
