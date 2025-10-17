/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
#import "keychain/ot/OctagonStateMachineHelpers.h"
#import "keychain/ckks/CKKSNearFutureScheduler.h"

NS_ASSUME_NONNULL_BEGIN

// An OctagonPendingFlag asks the state machine to add a flag in the future, when some conditions are met

// Currently, this is only time-based.
// Future planned conditions include "device is probably unlocked" and "device has network again"

typedef NS_OPTIONS(NSUInteger, OctagonPendingConditions) {
    OctagonPendingConditionsDeviceUnlocked = 1,
    OctagonPendingConditionsNetworkReachable = 2,
};

NSString* OctagonPendingConditionsToString(OctagonPendingConditions cond);

@interface OctagonPendingFlag : NSObject
@property (readonly) OctagonFlag* flag;

// NSDate after which this flag should become unpending
@property (nullable, readonly) NSDate* fireTime;

@property (readonly) OctagonPendingConditions conditions;

@property (nullable) NSOperation* afterOperation;

- (instancetype)initWithFlag:(OctagonFlag*)flag delayInSeconds:(NSTimeInterval)delay;
- (instancetype)initWithFlag:(OctagonFlag*)flag conditions:(OctagonPendingConditions)conditions;
- (instancetype)initWithFlag:(OctagonFlag*)flag after:(NSOperation*)op;

- (instancetype)initWithFlag:(OctagonFlag*)flag
                  conditions:(OctagonPendingConditions)conditions
              delayInSeconds:(NSTimeInterval)delay;

// The flag will unpend when the scheduler fires.

- (instancetype)initWithFlag:(OctagonFlag*)flag
                   scheduler:(CKKSNearFutureScheduler*)scheduler;

- (instancetype)initWithFlag:(OctagonFlag*)flag
                  conditions:(OctagonPendingConditions)conditions
                   scheduler:(CKKSNearFutureScheduler*)scheduler;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
