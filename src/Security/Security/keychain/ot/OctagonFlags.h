/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 30, 2023.
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

#import "keychain/ot/OctagonStateMachineHelpers.h"
#import "keychain/ot/OctagonPendingFlag.h"

NS_ASSUME_NONNULL_BEGIN

// OctagonFlags allow you to set binary flags for consumption by the state machine, similar to processor interrupts
// This allows the state machine to respond to external inputs or requests that don't need a timeout attached to them.
// Setting and removing flags are idempotent.

@protocol OctagonFlagContainer
- (BOOL)_onqueueContains:(OctagonFlag*)flag;
- (NSArray<NSString*>*)dumpFlags;
- (CKKSCondition*)conditionForFlag:(OctagonFlag*)flag;

- (CKKSCondition* _Nullable)conditionForFlagIfPresent:(OctagonFlag*)flag;
@end

@protocol OctagonFlagSetter <OctagonFlagContainer>
- (void)setFlag:(OctagonFlag*)flag;
- (void)_onqueueSetFlag:(OctagonFlag*)flag;
@end

@protocol OctagonFlagClearer <OctagonFlagSetter>
- (void)_onqueueRemoveFlag:(OctagonFlag*)flag;
@end

@interface OctagonFlags : NSObject <OctagonFlagContainer,
                                    OctagonFlagSetter,
                                    OctagonFlagClearer>

@property NSMutableDictionary<OctagonFlag*, CKKSCondition*>* flagConditions;

- (instancetype)initWithQueue:(dispatch_queue_t)queue
                        flags:(NSSet<OctagonFlag*>*)possibleFlags;

- (NSString*)contentsAsString;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
