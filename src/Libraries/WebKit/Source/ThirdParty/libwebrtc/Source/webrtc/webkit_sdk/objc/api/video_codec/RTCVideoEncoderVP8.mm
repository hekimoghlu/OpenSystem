/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 28, 2022.
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
#import <Foundation/Foundation.h>

#import "RTCVideoEncoderVP8.h"
#import "RTCWrappedNativeVideoEncoder.h"

#include "api/environment/environment_factory.h"
#include "modules/video_coding/codecs/vp8/include/vp8.h"

@implementation RTCVideoEncoderVP8

+ (id<RTCVideoEncoder>)vp8Encoder {
  return [[RTCWrappedNativeVideoEncoder alloc]
          initWithNativeEncoder:std::unique_ptr<webrtc::VideoEncoder>(webrtc::CreateVp8Encoder(webrtc::EnvironmentFactory().Create()))];
}

@end
