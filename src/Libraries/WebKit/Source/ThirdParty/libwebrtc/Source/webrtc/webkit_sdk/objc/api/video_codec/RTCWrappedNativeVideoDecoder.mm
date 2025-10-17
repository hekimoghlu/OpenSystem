/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 21, 2022.
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

#import "RTCWrappedNativeVideoDecoder.h"
#import "helpers/NSString+StdString.h"

@implementation RTCWrappedNativeVideoDecoder {
  std::unique_ptr<webrtc::VideoDecoder> _wrappedDecoder;
}

- (instancetype)initWithNativeDecoder:(std::unique_ptr<webrtc::VideoDecoder>)decoder {
  if (self = [super init]) {
    _wrappedDecoder = std::move(decoder);
  }

  return self;
}

- (std::unique_ptr<webrtc::VideoDecoder>)releaseWrappedDecoder {
  return std::move(_wrappedDecoder);
}

#pragma mark - RTCVideoDecoder

- (void)setCallback:(RTCVideoDecoderCallback)callback {
  RTC_DCHECK_NOTREACHED();
}

- (NSInteger)startDecodeWithNumberOfCores:(int)numberOfCores {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (NSInteger)releaseDecoder {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (NSInteger)decode:(RTCEncodedImage *)encodedImage
        missingFrames:(BOOL)missingFrames
    codecSpecificInfo:(nullable id<RTCCodecSpecificInfo>)info
         renderTimeMs:(int64_t)renderTimeMs {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (NSString *)implementationName {
#if !defined(WEBRTC_WEBKIT_BUILD)
  RTC_DCHECK_NOTREACHED();
#endif
  return nil;
}

@end
