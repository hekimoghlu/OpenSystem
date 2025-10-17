/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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
#ifndef SecItemRateLimit_tests_h
#define SecItemRateLimit_tests_h

#import "SecItemRateLimit.h"
#import <Foundation/Foundation.h>

// Broken out into header for testing convenience.
// If you need this, why?
@interface SecItemRateLimit : NSObject

@property (nonatomic, readonly) int roCapacity;
@property (nonatomic, readonly) double roRate;
@property (nonatomic, readonly) int rwCapacity;
@property (nonatomic, readonly) double rwRate;
@property (nonatomic, readonly) double limitMultiplier;

@property (nonatomic, readonly) NSDate* roBucket;
@property (nonatomic, readonly) NSDate* rwBucket;

- (bool)shouldCountAPICalls;
- (bool)isEnabled;
- (void)forceEnabled:(bool)force;

+ (instancetype)getStaticRateLimit;
+ (void)resetStaticRateLimit;

@end

#endif /* SecItemRateLimit_tests_h */
