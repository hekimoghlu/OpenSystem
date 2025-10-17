/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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

#import "RTCVideoEncoderVP9.h"
#import "RTCWrappedNativeVideoEncoder.h"

#include "api/environment/environment_factory.h"
#include "modules/video_coding/codecs/vp9/include/vp9.h"
#include "webkit_sdk/objc/api/peerconnection/RTCVideoCodecInfo+Private.h"

@implementation RTCVideoEncoderVP9

+ (id<RTCVideoEncoder>)vp9Encoder:(RTCVideoCodecInfo *)codecInfo {
  return [[RTCWrappedNativeVideoEncoder alloc]
      initWithNativeEncoder:std::unique_ptr<webrtc::VideoEncoder>(webrtc::CreateVp9Encoder(webrtc::EnvironmentFactory().Create(), { webrtc::VP9Profile::kProfile0 }))];
}

@end
