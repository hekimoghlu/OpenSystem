/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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

NS_ASSUME_NONNULL_BEGIN

/** Information for header. Corresponds to webrtc::RTPFragmentationHeader. */
RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCRtpFragmentationHeader")))
@interface RTCRtpFragmentationHeader : NSObject

@property(nonatomic, strong) NSArray<NSNumber *> *fragmentationOffset;
@property(nonatomic, strong) NSArray<NSNumber *> *fragmentationLength;
@property(nonatomic, strong) NSArray<NSNumber *> *fragmentationTimeDiff;
@property(nonatomic, strong) NSArray<NSNumber *> *fragmentationPlType;

@end

NS_ASSUME_NONNULL_END
