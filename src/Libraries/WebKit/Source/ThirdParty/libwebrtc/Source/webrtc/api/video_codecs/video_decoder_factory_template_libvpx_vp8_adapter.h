/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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
#ifndef API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_LIBVPX_VP8_ADAPTER_H_
#define API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_LIBVPX_VP8_ADAPTER_H_

#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "modules/video_coding/codecs/vp8/include/vp8.h"

namespace webrtc {
struct LibvpxVp8DecoderTemplateAdapter {
  static std::vector<SdpVideoFormat> SupportedFormats() {
    return {SdpVideoFormat::VP8()};
  }

  static std::unique_ptr<VideoDecoder> CreateDecoder(
      const Environment& env,
      const SdpVideoFormat& /* format */) {
    return CreateVp8Decoder(env);
  }
};
}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_LIBVPX_VP8_ADAPTER_H_
