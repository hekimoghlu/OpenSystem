/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#if OCTAGON

#import <Foundation/Foundation.h>
#import <dispatch/dispatch.h>
#import "keychain/ckks/CKKSResultOperation.h"

NS_ASSUME_NONNULL_BEGIN

/*
 * The CKKSNearFutureScheduler is intended to rate-limit an operation. When
 * triggered, it will schedule the operation to take place in the future.
 * Further triggers during the delay period will not cause the operation to
 * occur again, but they may cause the delay period to extend.
 *
 * Triggers after the delay period will start another delay period.
 */

@interface CKKSNearFutureScheduler : NSObject

@property (nullable, readonly) NSDate* nextFireTime;
@property void (^futureBlock)(void);

// Will execute every time futureBlock is called, just after the future block.
// Operations added in the futureBlock will receive the next operationDependency, so they won't run again until futureBlock occurs again.
@property (readonly) CKKSResultOperation* operationDependency;


// dependencyDescriptionCode will be integrated into the operationDependency as per the rules in CKKSResultOperation.h
- (instancetype)initWithName:(NSString*)name
                       delay:(dispatch_time_t)ns
            keepProcessAlive:(bool)keepProcessAlive
   dependencyDescriptionCode:(NSInteger)code
                       block:(void (^_Nonnull)(void))futureBlock;

- (instancetype)initWithName:(NSString*)name
                initialDelay:(dispatch_time_t)initialDelay
             continuingDelay:(dispatch_time_t)continuingDelay
            keepProcessAlive:(bool)keepProcessAlive
   dependencyDescriptionCode:(NSInteger)code
                       block:(void (^_Nonnull)(void))futureBlock;

- (instancetype)initWithName:(NSString*)name
                initialDelay:(dispatch_time_t)initialDelay
          exponentialBackoff:(double)backoff
                maximumDelay:(dispatch_time_t)maximumDelay
            keepProcessAlive:(bool)keepProcessAlive
   dependencyDescriptionCode:(NSInteger)code
                       block:(void (^_Nonnull)(void))futureBlock;


- (instancetype)initWithName:(NSString*)name
                initialDelay:(dispatch_time_t)initialDelay
          exponentialBackoff:(double)backoff
                maximumDelay:(dispatch_time_t)maximumDelay
            keepProcessAlive:(bool)keepProcessAlive
   dependencyDescriptionCode:(NSInteger)code
                    qosClass:(qos_class_t)qosClass
                       block:(void (^_Nonnull)(void))futureBlock;

- (void)trigger;

- (void)cancel;

// Don't trigger again until at least this much time has passed.
- (void)waitUntil:(uint64_t)delay;

// Trigger at this time (unless further instructions are given)
- (void)triggerAt:(uint64_t)delay;

- (void)changeDelays:(dispatch_time_t)initialDelay continuingDelay:(dispatch_time_t)continuingDelay;

// tests
@property CKKSCondition* liveRequestReceived;

@end

NS_ASSUME_NONNULL_END
#endif // OCTAGON
