/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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

RTC_OBJC_EXPORT extern NSString *const kRTCVideoCodecH264Name;
RTC_OBJC_EXPORT extern NSString *const kRTCLevel31ConstrainedHigh;
RTC_OBJC_EXPORT extern NSString *const kRTCLevel31ConstrainedBaseline;
RTC_OBJC_EXPORT extern NSString *const kRTCMaxSupportedH264ProfileLevelConstrainedHigh;
RTC_OBJC_EXPORT extern NSString *const kRTCMaxSupportedH264ProfileLevelConstrainedBaseline;

/** H264 Profiles and levels. */
typedef NS_ENUM(NSUInteger, RTCH264Profile) {
  RTCH264ProfileConstrainedBaseline,
  RTCH264ProfileBaseline,
  RTCH264ProfileMain,
  RTCH264ProfileConstrainedHigh,
  RTCH264ProfileHigh,
};

typedef NS_ENUM(NSUInteger, RTCH264Level) {
  RTCH264Level1_b = 0,
  RTCH264Level1 = 10,
  RTCH264Level1_1 = 11,
  RTCH264Level1_2 = 12,
  RTCH264Level1_3 = 13,
  RTCH264Level2 = 20,
  RTCH264Level2_1 = 21,
  RTCH264Level2_2 = 22,
  RTCH264Level3 = 30,
  RTCH264Level3_1 = 31,
  RTCH264Level3_2 = 32,
  RTCH264Level4 = 40,
  RTCH264Level4_1 = 41,
  RTCH264Level4_2 = 42,
  RTCH264Level5 = 50,
  RTCH264Level5_1 = 51,
  RTCH264Level5_2 = 52
};

RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCH264ProfileLevelId")))
@interface RTCH264ProfileLevelId : NSObject

@property(nonatomic, readonly) RTCH264Profile profile;
@property(nonatomic, readonly) RTCH264Level level;
@property(nonatomic, readonly) NSString *hexString;

- (instancetype)initWithHexString:(NSString *)hexString;
- (instancetype)initWithProfile:(RTCH264Profile)profile level:(RTCH264Level)level;

@end
