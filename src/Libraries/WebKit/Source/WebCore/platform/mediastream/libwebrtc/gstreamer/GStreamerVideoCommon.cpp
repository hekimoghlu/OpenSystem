/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#if ENABLE(VIDEO) && ENABLE(MEDIA_STREAM) && USE(LIBWEBRTC) && USE(GSTREAMER)
#include "GStreamerVideoCommon.h"

#include "absl/types/optional.h"
#include "webrtc/api/video_codecs/h264_profile_level_id.h"
#include "webrtc/media/base/codec.h"

namespace WebCore {

static webrtc::SdpVideoFormat createH264Format(webrtc::H264Profile profile, webrtc::H264Level level, const std::string& packetizationMode)
{
    const auto profileString = webrtc::H264ProfileLevelIdToString(webrtc::H264ProfileLevelId(profile, level));

    return webrtc::SdpVideoFormat(cricket::kH264CodecName,
        { { cricket::kH264FmtpProfileLevelId, *profileString },
            { cricket::kH264FmtpLevelAsymmetryAllowed, "1" },
            { cricket::kH264FmtpPacketizationMode, packetizationMode } });
}

std::vector<webrtc::SdpVideoFormat> supportedH264Formats()
{
    // @TODO Create from encoder src pad caps template
    //
    // We only support encoding Constrained Baseline Profile (CBP), but the decoder supports more
    // profiles. We can list all profiles here that are supported by the decoder and that are also
    // supersets of CBP, i.e. the decoder for that profile is required to be able to decode
    // CBP. This means we can encode and send CBP even though we negotiated a potentially higher
    // profile. See the H264 spec for more information.
    //
    // We support both packetization modes 0 (mandatory) and 1 (optional, preferred).
    return {
        createH264Format(webrtc::H264Profile::kProfileBaseline, webrtc::H264Level::kLevel3_1, "1"),
        createH264Format(webrtc::H264Profile::kProfileBaseline, webrtc::H264Level::kLevel3_1, "0"),
        createH264Format(webrtc::H264Profile::kProfileConstrainedBaseline, webrtc::H264Level::kLevel3_1, "1"),
        createH264Format(webrtc::H264Profile::kProfileConstrainedBaseline, webrtc::H264Level::kLevel3_1, "0"),
    };
}

} // namespace WebCore

#endif
