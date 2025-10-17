/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#ifndef API_VIDEO_CODECS_LIBAOM_AV1_ENCODER_FACTORY_H_
#define API_VIDEO_CODECS_LIBAOM_AV1_ENCODER_FACTORY_H_

#include <map>
#include <memory>
#include <string>

#include "api/video_codecs/video_encoder_factory_interface.h"
#include "api/video_codecs/video_encoder_interface.h"

namespace webrtc {
class LibaomAv1EncoderFactory final : VideoEncoderFactoryInterface {
 public:
  std::string CodecName() const override;
  std::string ImplementationName() const override;
  std::map<std::string, std::string> CodecSpecifics() const override;

  Capabilities GetEncoderCapabilities() const override;
  std::unique_ptr<VideoEncoderInterface> CreateEncoder(
      const StaticEncoderSettings& settings,
      const std::map<std::string, std::string>& encoder_specific_settings)
      override;
};
}  // namespace webrtc
#endif  // API_VIDEO_CODECS_LIBAOM_AV1_ENCODER_FACTORY_H_
