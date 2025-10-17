/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 7, 2023.
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

typedef NS_ENUM(NSUInteger, RTCVideoCodecMode) {
  RTCVideoCodecModeRealtimeVideo,
  RTCVideoCodecModeScreensharing,
};

/** Settings for encoder. Corresponds to webrtc::VideoCodec. */
RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCVideoEncoderSettings")))
@interface RTCVideoEncoderSettings : NSObject

@property(nonatomic, strong) NSString *name;

@property(nonatomic, assign) unsigned short width;
@property(nonatomic, assign) unsigned short height;

@property(nonatomic, assign) unsigned int startBitrate;  // kilobits/sec.
@property(nonatomic, assign) unsigned int maxBitrate;
@property(nonatomic, assign) unsigned int minBitrate;

@property(nonatomic, assign) uint32_t maxFramerate;

@property(nonatomic, assign) unsigned int qpMax;
@property(nonatomic, assign) RTCVideoCodecMode mode;

@end

NS_ASSUME_NONNULL_END
