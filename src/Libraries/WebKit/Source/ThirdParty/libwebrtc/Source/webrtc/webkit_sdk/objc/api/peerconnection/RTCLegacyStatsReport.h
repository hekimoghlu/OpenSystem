/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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

/** This does not currently conform to the spec. */
RTC_OBJC_EXPORT
@interface RTCLegacyStatsReport : NSObject

/** Time since 1970-01-01T00:00:00Z in milliseconds. */
@property(nonatomic, readonly) CFTimeInterval timestamp;

/** The type of stats held by this object. */
@property(nonatomic, readonly) NSString *type;

/** The identifier for this object. */
@property(nonatomic, readonly) NSString *reportId;

/** A dictionary holding the actual stats. */
@property(nonatomic, readonly) NSDictionary<NSString *, NSString *> *values;

- (instancetype)init NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
