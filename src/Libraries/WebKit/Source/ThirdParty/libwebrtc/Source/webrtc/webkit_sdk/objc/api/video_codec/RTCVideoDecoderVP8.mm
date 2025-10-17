/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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

#import "RTCVideoDecoderVP8.h"
#import "RTCWrappedNativeVideoDecoder.h"

#include "api/environment/environment_factory.h"
#include "modules/video_coding/codecs/vp8/include/vp8.h"

@implementation RTCVideoDecoderVP8

+ (id<RTCVideoDecoder>)vp8Decoder {
  return [[RTCWrappedNativeVideoDecoder alloc]
          initWithNativeDecoder:std::unique_ptr<webrtc::VideoDecoder>(webrtc::CreateVp8Decoder(webrtc::EnvironmentFactory().Create()))];
}

@end
