/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 8, 2025.
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
#ifndef SDK_OBJC_NATIVE_SRC_OBJC_VIDEO_ENCODER_FACTORY_H_
#define SDK_OBJC_NATIVE_SRC_OBJC_VIDEO_ENCODER_FACTORY_H_

#import <Foundation/Foundation.h>

#include "api/video_codecs/video_encoder_factory.h"

@protocol RTCVideoEncoderFactory;

namespace webrtc {

class ObjCVideoEncoderFactory : public VideoEncoderFactory {
 public:
  explicit ObjCVideoEncoderFactory(id<RTCVideoEncoderFactory>);
  ~ObjCVideoEncoderFactory() override;

  id<RTCVideoEncoderFactory> wrapped_encoder_factory() const;

  std::vector<SdpVideoFormat> GetSupportedFormats() const override;
  std::vector<SdpVideoFormat> GetImplementations() const override;
  std::unique_ptr<VideoEncoder> Create(
      const Environment& env,
      const SdpVideoFormat& format) override;

 private:
  id<RTCVideoEncoderFactory> encoder_factory_;
};

}  // namespace webrtc

#endif  // SDK_OBJC_NATIVE_SRC_OBJC_VIDEO_ENCODER_FACTORY_H_
