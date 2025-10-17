/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#ifndef API_VIDEO_CODECS_VIDEO_DECODER_SOFTWARE_FALLBACK_WRAPPER_H_
#define API_VIDEO_CODECS_VIDEO_DECODER_SOFTWARE_FALLBACK_WRAPPER_H_

#include <memory>

#include "api/environment/environment.h"
#include "api/video_codecs/video_decoder.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// Used to wrap external VideoDecoders to provide a fallback option on
// software decoding when a hardware decoder fails to decode a stream due to
// hardware restrictions, such as max resolution.
RTC_EXPORT std::unique_ptr<VideoDecoder>
CreateVideoDecoderSoftwareFallbackWrapper(
    const Environment& env,
    std::unique_ptr<VideoDecoder> sw_fallback_decoder,
    std::unique_ptr<VideoDecoder> hw_decoder);

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_DECODER_SOFTWARE_FALLBACK_WRAPPER_H_
