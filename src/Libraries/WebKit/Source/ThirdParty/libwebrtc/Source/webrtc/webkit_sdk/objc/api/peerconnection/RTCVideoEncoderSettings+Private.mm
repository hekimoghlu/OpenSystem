/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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
#import "RTCVideoEncoderSettings+Private.h"

#import "helpers/NSString+StdString.h"

@implementation RTCVideoEncoderSettings (Private)

- (instancetype)initWithNativeVideoCodec:(const webrtc::VideoCodec *)videoCodec {
  if (self = [super init]) {
    if (videoCodec) {
      const char *codecName = CodecTypeToPayloadString(videoCodec->codecType);
      self.name = [NSString stringWithUTF8String:codecName];

      self.width = videoCodec->width;
      self.height = videoCodec->height;
      self.startBitrate = videoCodec->startBitrate;
      self.maxBitrate = videoCodec->maxBitrate;
      self.minBitrate = videoCodec->minBitrate;
      self.maxFramerate = videoCodec->maxFramerate;
      self.qpMax = videoCodec->qpMax;
      self.mode = (RTCVideoCodecMode)videoCodec->mode;
    }
  }

  return self;
}

- (webrtc::VideoCodec)nativeVideoCodec {
  webrtc::VideoCodec videoCodec;
  videoCodec.width = self.width;
  videoCodec.height = self.height;
  videoCodec.startBitrate = self.startBitrate;
  videoCodec.maxBitrate = self.maxBitrate;
  videoCodec.minBitrate = self.minBitrate;
  videoCodec.maxBitrate = self.maxBitrate;
  videoCodec.qpMax = self.qpMax;
  videoCodec.mode = (webrtc::VideoCodecMode)self.mode;

  return videoCodec;
}

@end
