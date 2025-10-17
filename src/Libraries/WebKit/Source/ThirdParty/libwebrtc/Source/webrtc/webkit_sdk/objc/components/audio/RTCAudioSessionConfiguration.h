/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 27, 2023.
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
#import <AVFoundation/AVFoundation.h>
#import <Foundation/Foundation.h>

#import "RTCMacros.h"

NS_ASSUME_NONNULL_BEGIN

RTC_EXTERN const int kRTCAudioSessionPreferredNumberOfChannels;
RTC_EXTERN const double kRTCAudioSessionHighPerformanceSampleRate;
RTC_EXTERN const double kRTCAudioSessionLowComplexitySampleRate;
RTC_EXTERN const double kRTCAudioSessionHighPerformanceIOBufferDuration;
RTC_EXTERN const double kRTCAudioSessionLowComplexityIOBufferDuration;

// Struct to hold configuration values.
RTC_OBJC_EXPORT
@interface RTCAudioSessionConfiguration : NSObject

@property(nonatomic, strong) NSString *category;
@property(nonatomic, assign) AVAudioSessionCategoryOptions categoryOptions;
@property(nonatomic, strong) NSString *mode;
@property(nonatomic, assign) double sampleRate;
@property(nonatomic, assign) NSTimeInterval ioBufferDuration;
@property(nonatomic, assign) NSInteger inputNumberOfChannels;
@property(nonatomic, assign) NSInteger outputNumberOfChannels;

/** Initializes configuration to defaults. */
- (instancetype)init NS_DESIGNATED_INITIALIZER;

/** Returns the current configuration of the audio session. */
+ (instancetype)currentConfiguration;
/** Returns the configuration that WebRTC needs. */
+ (instancetype)webRTCConfiguration;
/** Provide a way to override the default configuration. */
+ (void)setWebRTCConfiguration:(RTCAudioSessionConfiguration *)configuration;

@end

NS_ASSUME_NONNULL_END
