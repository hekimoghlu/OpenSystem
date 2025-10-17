/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

#import "RTCMacros.h"
#import "RTCMediaSource.h"
#import "RTCVideoCapturer.h"

NS_ASSUME_NONNULL_BEGIN

RTC_OBJC_EXPORT

@interface RTCVideoSource : RTCMediaSource <RTCVideoCapturerDelegate>

- (instancetype)init NS_UNAVAILABLE;

/**
 * Calling this function will cause frames to be scaled down to the
 * requested resolution. Also, frames will be cropped to match the
 * requested aspect ratio, and frames will be dropped to match the
 * requested fps. The requested aspect ratio is orientation agnostic and
 * will be adjusted to maintain the input orientation, so it doesn't
 * matter if e.g. 1280x720 or 720x1280 is requested.
 */
- (void)adaptOutputFormatToWidth:(int)width height:(int)height fps:(int)fps;

@end

NS_ASSUME_NONNULL_END
