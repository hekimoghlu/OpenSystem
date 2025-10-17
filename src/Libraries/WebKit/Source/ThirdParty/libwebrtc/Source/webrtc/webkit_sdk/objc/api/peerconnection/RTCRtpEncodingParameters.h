/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 24, 2022.
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

/** Corresponds to webrtc::Priority. */
typedef NS_ENUM(NSInteger, RTCPriority) {
  RTCPriorityVeryLow,
  RTCPriorityLow,
  RTCPriorityMedium,
  RTCPriorityHigh
};

RTC_OBJC_EXPORT
__attribute__((objc_runtime_name("WK_RTCRtpEncodingParameters")))
@interface RTCRtpEncodingParameters : NSObject

/** The idenfifier for the encoding layer. This is used in simulcast. */
@property(nonatomic, copy, nullable) NSString *rid;

/** Controls whether the encoding is currently transmitted. */
@property(nonatomic, assign) BOOL isActive;

/** The maximum bitrate to use for the encoding, or nil if there is no
 *  limit.
 */
@property(nonatomic, copy, nullable) NSNumber *maxBitrateBps;

/** The minimum bitrate to use for the encoding, or nil if there is no
 *  limit.
 */
@property(nonatomic, copy, nullable) NSNumber *minBitrateBps;

/** The maximum framerate to use for the encoding, or nil if there is no
 *  limit.
 */
@property(nonatomic, copy, nullable) NSNumber *maxFramerate;

/** The requested number of temporal layers to use for the encoding, or nil
 * if the default should be used.
 */
@property(nonatomic, copy, nullable) NSNumber *numTemporalLayers;

/** Scale the width and height down by this factor for video. If nil,
 * implementation default scaling factor will be used.
 */
@property(nonatomic, copy, nullable) NSNumber *scaleResolutionDownBy;

/** The SSRC being used by this encoding. */
@property(nonatomic, readonly, nullable) NSNumber *ssrc;

/** The relative DiffServ Code Point priority. */
@property(nonatomic, assign) RTCPriority networkPriority;

- (instancetype)init NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
