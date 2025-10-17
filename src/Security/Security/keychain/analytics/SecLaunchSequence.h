/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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
#if __OBJC__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*
 *  Takes a sequence of events and report their time relative from the starting point
 *  Duplicate events are counted.
 */

@interface SecLaunchSequence : NSObject
@property (readonly) bool launched;
@property (assign) bool firstLaunch;
@property (readonly) NSString *name;

- (instancetype)init NS_UNAVAILABLE;

// name should be dns reverse notation, com.apple.label
- (instancetype)initWithRocketName:(NSString *)name;

// value must be a valid JSON compatible type
- (void)addAttribute:(NSString *)key value:(id)value;
- (void)addEvent:(NSString *)eventname;

- (void)launch;

- (void)addDependantLaunch:(NSString *)name child:(SecLaunchSequence *)child;

- (NSArray *) eventsRelativeTime;
- (NSDictionary<NSString*,id>* _Nullable) metricsReport;

// For including in human readable diagnostics
- (NSArray<NSString *> *)eventsByTime;
@end

NS_ASSUME_NONNULL_END

#endif /* __OBJC__ */
