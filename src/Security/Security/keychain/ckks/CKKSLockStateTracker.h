/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 11, 2023.
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

@protocol CKKSLockStateNotification <NSObject>
- (void)lockStateChangeNotification:(bool)unlocked;
@end

NS_ASSUME_NONNULL_BEGIN

@protocol CKKSLockStateProviderProtocol
- (BOOL)queryAKSLocked;
@end

@interface CKKSLockStateTracker : NSObject
@property (nullable) NSOperation* unlockDependency;
@property (readonly) bool isLocked;

@property (readonly,nullable) NSDate* lastUnlockTime;

@property id<CKKSLockStateProviderProtocol> lockStateProvider;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithProvider:(id<CKKSLockStateProviderProtocol>)provider;

// Force a recheck of the keybag lock state
- (void)recheck;

// Check if this error code is related to keybag is locked and we should retry later
- (bool)isLockedError:(NSError*)error;

- (void)addLockStateObserver:(id<CKKSLockStateNotification>)object;

// Call this to get a CKKSLockStateTracker to use. This tracker will likely be tracking real AKS.
+ (CKKSLockStateTracker*)globalTracker;
@end

@interface CKKSActualLockStateProvider : NSObject <CKKSLockStateProviderProtocol>
- (instancetype)init;
@end

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
