/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
#ifndef API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_H_
#define API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_H_

#include <memory>
#include <vector>

#include "api/environment/environment.h"
#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_decoder.h"
#include "rtc_base/system/rtc_export.h"

namespace webrtc {

// A factory that creates VideoDecoders.
// NOTE: This class is still under development and may change without notice.
class RTC_EXPORT VideoDecoderFactory {
 public:
  struct CodecSupport {
    bool is_supported = false;
    bool is_power_efficient = false;
  };

  virtual ~VideoDecoderFactory() = default;

  // Returns a list of supported video formats in order of preference, to use
  // for signaling etc.
  virtual std::vector<SdpVideoFormat> GetSupportedFormats() const = 0;

  // Query whether the specifed format is supported or not and if it will be
  // power efficient, which is currently interpreted as if there is support for
  // hardware acceleration.
  // The parameter `reference_scaling` is used to query support for prediction
  // across spatial layers. An example where support for reference scaling is
  // needed is if the video stream is produced with a scalability mode that has
  // a dependency between the spatial layers. See
  // https://w3c.github.io/webrtc-svc/#scalabilitymodes* for a specification of
  // different scalabilty modes. NOTE: QueryCodecSupport is currently an
  // experimental feature that is subject to change without notice.
  virtual CodecSupport QueryCodecSupport(const SdpVideoFormat& format,
                                         bool reference_scaling) const;

  // Creates a VideoDecoder for the specified `format`.
  virtual std::unique_ptr<VideoDecoder> Create(
      const Environment& env,
      const SdpVideoFormat& format) = 0;
};

}  // namespace webrtc

#endif  // API_VIDEO_CODECS_VIDEO_DECODER_FACTORY_H_
