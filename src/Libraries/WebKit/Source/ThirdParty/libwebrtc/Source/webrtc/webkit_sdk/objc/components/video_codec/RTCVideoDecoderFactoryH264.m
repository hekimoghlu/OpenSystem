/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#import "RTCVideoDecoderFactoryH264.h"

#import "RTCH264ProfileLevelId.h"
#import "RTCVideoDecoderH264.h"

@implementation RTCVideoDecoderFactoryH264

- (NSArray<RTCVideoCodecInfo *> *)supportedCodecs {
  NSMutableArray<RTCVideoCodecInfo *> *codecs = [NSMutableArray array];
  NSString *codecName = kRTCVideoCodecH264Name;

  NSDictionary<NSString *, NSString *> *constrainedHighParams = @{
    @"profile-level-id" : kRTCMaxSupportedH264ProfileLevelConstrainedHigh,
    @"level-asymmetry-allowed" : @"1",
    @"packetization-mode" : @"1",
  };
  RTCVideoCodecInfo *constrainedHighInfo =
      [[RTCVideoCodecInfo alloc] initWithName:codecName parameters:constrainedHighParams];
  [codecs addObject:constrainedHighInfo];

  NSDictionary<NSString *, NSString *> *constrainedBaselineParams = @{
    @"profile-level-id" : kRTCMaxSupportedH264ProfileLevelConstrainedBaseline,
    @"level-asymmetry-allowed" : @"1",
    @"packetization-mode" : @"1",
  };
  RTCVideoCodecInfo *constrainedBaselineInfo =
      [[RTCVideoCodecInfo alloc] initWithName:codecName parameters:constrainedBaselineParams];
  [codecs addObject:constrainedBaselineInfo];

  return [codecs copy];
}

- (id<RTCVideoDecoder>)createDecoder:(RTCVideoCodecInfo *)info {
  return [[RTCVideoDecoderH264 alloc] init];
}

@end
