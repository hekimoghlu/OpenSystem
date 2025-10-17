/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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

#import "RTCWrappedNativeVideoEncoder.h"
#import "helpers/NSString+StdString.h"

@implementation RTCWrappedNativeVideoEncoder {
  std::unique_ptr<webrtc::VideoEncoder> _wrappedEncoder;
}

- (instancetype)initWithNativeEncoder:(std::unique_ptr<webrtc::VideoEncoder>)encoder {
  if (self = [super init]) {
    _wrappedEncoder = std::move(encoder);
  }

  return self;
}

- (std::unique_ptr<webrtc::VideoEncoder>)releaseWrappedEncoder {
  return std::move(_wrappedEncoder);
}

#pragma mark - RTCVideoEncoder

- (void)setCallback:(RTCVideoEncoderCallback)callback {
  RTC_DCHECK_NOTREACHED();
}

- (NSInteger)startEncodeWithSettings:(RTCVideoEncoderSettings *)settings
                       numberOfCores:(int)numberOfCores {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (NSInteger)releaseEncoder {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (NSInteger)encode:(RTCVideoFrame *)frame
    codecSpecificInfo:(nullable id<RTCCodecSpecificInfo>)info
           frameTypes:(NSArray<NSNumber *> *)frameTypes {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (int)setBitrate:(uint32_t)bitrateKbit framerate:(uint32_t)framerate {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (NSString *)implementationName {
  RTC_DCHECK_NOTREACHED();
  return nil;
}

- (nullable RTCVideoEncoderQpThresholds *)scalingSettings {
  RTC_DCHECK_NOTREACHED();
  return nil;
}

@end
