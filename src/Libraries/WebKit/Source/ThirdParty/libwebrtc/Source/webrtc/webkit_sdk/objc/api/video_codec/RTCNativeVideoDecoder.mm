/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

#import "RTCNativeVideoDecoder.h"
#import "base/RTCMacros.h"
#import "helpers/NSString+StdString.h"
#include "rtc_base/checks.h"

@implementation RTC_OBJC_TYPE (RTCNativeVideoDecoder)

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

// TODO: bugs.webrtc.org/15444 - Remove obsolete missingFrames param.
- (NSInteger)decode:(RTC_OBJC_TYPE(RTCEncodedImage) *)encodedImage
        missingFrames:(BOOL)missingFrames
    codecSpecificInfo:(nullable id<RTC_OBJC_TYPE(RTCCodecSpecificInfo)>)info
         renderTimeMs:(int64_t)renderTimeMs {
  RTC_DCHECK_NOTREACHED();
  return 0;
}

- (NSString *)implementationName {
  RTC_DCHECK_NOTREACHED();
  return nil;
}

@end
