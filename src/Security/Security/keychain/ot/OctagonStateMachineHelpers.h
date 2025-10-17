/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 8, 2024.
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
#import "keychain/ckks/CKKSResultOperation.h"
#import "keychain/ckks/CKKSGroupOperation.h"
#import "keychain/ckks/CKKSAnalytics.h"

NS_ASSUME_NONNULL_BEGIN

@protocol OctagonStateString <NSObject>
@end
typedef NSString<OctagonStateString> OctagonState;

@protocol OctagonFlagString <NSObject>
@end
typedef NSString<OctagonFlagString> OctagonFlag;

// NotStarted indicates that this state machine is not yet started
extern OctagonState* const OctagonStateMachineNotStarted;

// Halted indicates that the state machine is halted, and won't move again
extern OctagonState* const OctagonStateMachineHalted;

@protocol OctagonStateTransitionOperationProtocol
// Holds this operation's opinion of the next state, given that this operation just ran
@property OctagonState* nextState;

// Hold the state this operation was originally hoping to enter
@property (readonly) OctagonState* intendedState;
@end


@interface OctagonStateTransitionOperation : CKKSResultOperation <OctagonStateTransitionOperationProtocol>
@property OctagonState* nextState;
@property (readonly) OctagonState* intendedState;

+ (instancetype)named:(NSString*)name
            intending:(OctagonState*)intendedState
           errorState:(OctagonState*)errorState
  withBlockTakingSelf:(void(^)(OctagonStateTransitionOperation* op))block;

// convenience constructor. Will always succeed at entering the state.
+ (instancetype)named:(NSString*)name
             entering:(OctagonState*)intendedState NS_SWIFT_NAME(init(name:entering:));
@end

// Just like OctagonStateTransitionOperation, but as a Group Operation
@interface OctagonStateTransitionGroupOperation : CKKSGroupOperation <OctagonStateTransitionOperationProtocol>
@property OctagonState* nextState;
@property (readonly) OctagonState* intendedState;

+ (instancetype)named:(NSString*)name
            intending:(OctagonState*)intendedState
           errorState:(OctagonState*)errorState
  withBlockTakingSelf:(void(^)(OctagonStateTransitionGroupOperation* op))block;
@end


@interface OctagonStateTransitionRequest<__covariant OperationType : CKKSResultOperation<OctagonStateTransitionOperationProtocol>*> : NSObject
@property (readonly) NSString* name;
@property (readonly) NSSet<OctagonState*>* sourceStates;
@property (readonly) OperationType transitionOperation;

- (OperationType _Nullable)_onqueueStart;
- (void)onqueueHandleStartTimeout:(NSError*)stateMachineStateError;

- (instancetype)init:(NSString*)name
        sourceStates:(NSSet<OctagonState*>*)sourceStates
         serialQueue:(dispatch_queue_t)queue
        transitionOp:(OperationType)transitionOp;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
