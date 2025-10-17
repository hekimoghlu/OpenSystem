/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#include "api/video_codecs/builtin_video_encoder_factory.h"

#include <memory>
#include <string>

#include "api/video_codecs/sdp_video_format.h"
#include "api/video_codecs/video_encoder_factory.h"
#include "test/gtest.h"

namespace webrtc {

TEST(BuiltinVideoEncoderFactoryTest, AnnouncesVp9AccordingToBuildFlags) {
  std::unique_ptr<VideoEncoderFactory> factory =
      CreateBuiltinVideoEncoderFactory();
  bool claims_vp9_support = false;
  for (const SdpVideoFormat& format : factory->GetSupportedFormats()) {
    if (format.name == "VP9") {
      claims_vp9_support = true;
      break;
    }
  }
#if defined(RTC_ENABLE_VP9)
  EXPECT_TRUE(claims_vp9_support);
#else
  EXPECT_FALSE(claims_vp9_support);
#endif  // defined(RTC_ENABLE_VP9)
}

}  // namespace webrtc
