/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 30, 2021.
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
#ifndef API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_DAV1D_ADAPTER_H_
#define API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_DAV1D_ADAPTER_H_

#include <memory>
#include <vector>

#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "modules/video_coding/codecs/av1/dav1d_decoder.h"

namespace webrtc {
struct Dav1dDecoderTemplateAdapter {
  static std::vector<SdpVideoFormat> SupportedFormats() {
    return {SdpVideoFormat::AV1Profile0(), SdpVideoFormat::AV1Profile1()};
  }

  static std::unique_ptr<VideoDecoder> CreateDecoder(
      const SdpVideoFormat& /* format */) {
    return CreateDav1dDecoder();
  }
};

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_TEMPLATE_DAV1D_ADAPTER_H_
