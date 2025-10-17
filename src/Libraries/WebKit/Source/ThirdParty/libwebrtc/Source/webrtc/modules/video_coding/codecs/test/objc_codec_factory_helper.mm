/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 14, 2024.
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
#include "modules/video_coding/codecs/test/objc_codec_factory_helper.h"

#import "sdk/objc/components/video_codec/RTCVideoDecoderFactoryH264.h"
#import "sdk/objc/components/video_codec/RTCVideoEncoderFactoryH264.h"
#include "sdk/objc/native/api/video_decoder_factory.h"
#include "sdk/objc/native/api/video_encoder_factory.h"

namespace webrtc {
namespace test {

std::unique_ptr<VideoEncoderFactory> CreateObjCEncoderFactory() {
  return ObjCToNativeVideoEncoderFactory([[RTC_OBJC_TYPE(RTCVideoEncoderFactoryH264) alloc] init]);
}

std::unique_ptr<VideoDecoderFactory> CreateObjCDecoderFactory() {
  return ObjCToNativeVideoDecoderFactory([[RTC_OBJC_TYPE(RTCVideoDecoderFactoryH264) alloc] init]);
}

}  // namespace test
}  // namespace webrtc
