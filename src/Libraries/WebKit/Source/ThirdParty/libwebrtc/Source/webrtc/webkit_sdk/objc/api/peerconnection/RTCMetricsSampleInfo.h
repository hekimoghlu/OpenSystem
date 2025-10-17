/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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

RTC_OBJC_EXPORT
@interface RTCMetricsSampleInfo : NSObject

/**
 * Example of RTCMetricsSampleInfo:
 * name: "WebRTC.Video.InputFramesPerSecond"
 * min: 1
 * max: 100
 * bucketCount: 50
 * samples: [29]:2 [30]:1
 */

/** The name of the histogram. */
@property(nonatomic, readonly) NSString *name;

/** The minimum bucket value. */
@property(nonatomic, readonly) int min;

/** The maximum bucket value. */
@property(nonatomic, readonly) int max;

/** The number of buckets. */
@property(nonatomic, readonly) int bucketCount;

/** A dictionary holding the samples <value, # of events>. */
@property(nonatomic, readonly) NSDictionary<NSNumber *, NSNumber *> *samples;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
