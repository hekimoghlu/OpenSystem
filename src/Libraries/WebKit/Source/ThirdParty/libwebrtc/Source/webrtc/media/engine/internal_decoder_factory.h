/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#ifndef MEDIA_ENGINE_INTERNAL_DECODER_FACTORY_H_
#define MEDIA_ENGINE_INTERNAL_DECODER_FACTORY_H_

#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "api/video_codecs/video_decoder_factory.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

class RTC_EXPORT InternalDecoderFactory : public VideoDecoderFactory {
 public:
  std::vector<SdpVideoFormat> GetSupportedFormats() const override;
  CodecSupport QueryCodecSupport(const SdpVideoFormat& format,
                                 bool reference_scaling) const override;
  std::unique_ptr<VideoDecoder> Create(const Environment& env,
                                       const SdpVideoFormat& format) override;
};

}  // namespace webrtc

#endif  // MEDIA_ENGINE_INTERNAL_DECODER_FACTORY_H_
