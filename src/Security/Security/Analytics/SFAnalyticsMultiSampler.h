/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 13, 2024.
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
#if __OBJC2__
#ifndef SFAnalyticsMultiSampler_h
#define SFAnalyticsMultiSampler_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface SFAnalyticsMultiSampler : NSObject

@property (nonatomic) NSTimeInterval samplingInterval;
@property (nonatomic, readonly) NSString* name;
@property (nonatomic, readonly) BOOL oncePerReport;

- (instancetype)init NS_UNAVAILABLE;
- (NSDictionary<NSString*, NSNumber*>*)sampleNow;
- (void)pauseSampling;
- (void)resumeSampling;

@end

NS_ASSUME_NONNULL_END

#endif
#endif
