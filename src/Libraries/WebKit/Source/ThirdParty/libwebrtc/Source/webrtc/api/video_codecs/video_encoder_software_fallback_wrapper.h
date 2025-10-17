/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 28, 2024.
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
#ifndef API_VIDEO_CODECS_VIDEO_ENCODER_SOFTWARE_FALLBACK_WRAPPER_H_
#define API_VIDEO_CODECS_VIDEO_ENCODER_SOFTWARE_FALLBACK_WRAPPER_H_

#include <memory>

#include "api/environment/environment.h"
#include "api/video_codecs/video_encoder.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Used to wrap external VideoEncoders to provide a fallback option on
// software encoding when a hardware encoder fails to encode a stream due to
// hardware restrictions, such as max resolution.
// |bool prefer_temporal_support| indicates that if the software fallback
// encoder supports temporal layers but the hardware encoder does not, a
// fallback should be forced even if the encoder otherwise works.
RTC_EXPORT std::unique_ptr<VideoEncoder>
CreateVideoEncoderSoftwareFallbackWrapper(
    const Environment& env,
    std::unique_ptr<VideoEncoder> sw_fallback_encoder,
    std::unique_ptr<VideoEncoder> hw_encoder,
    bool prefer_temporal_support);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_ENCODER_SOFTWARE_FALLBACK_WRAPPER_H_
