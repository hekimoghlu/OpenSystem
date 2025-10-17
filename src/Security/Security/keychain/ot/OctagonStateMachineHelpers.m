/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#import "keychain/ot/OTStates.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/ot/OTDefines.h"
#import "keychain/ot/OTConstants.h"
#import "keychain/categories/NSError+UsefulConstructors.h"
#import "keychain/ckks/CloudKitCategories.h"
#import "keychain/ckks/CKKS.h"

OctagonState* const OctagonStateMachineNotStarted = (OctagonState*) @"not_started";
OctagonState* const OctagonStateMachineHalted = (OctagonState*) @"halted";

#pragma mark -- OctagonStateTransitionOperation

@implementation OctagonStateTransitionOperation
- (instancetype)initIntending:(OctagonState*)intendedState
                   errorState:(OctagonState*)errorState
{
    if((self = [super init])) {
        _nextState = errorState;
        _intendedState = intendedState;
    }
    return self;
}

- (NSString*)description
{
    return [NSString stringWithFormat:@"<OctagonStateTransitionOperation(%@): intended:%@ actual:%@>", self.name, self.intendedState, self.nextState];
}

+ (instancetype)named:(NSString*)name
            intending:(OctagonState*)intendedState
           errorState:(OctagonState*)errorState
  withBlockTakingSelf:(void(^)(OctagonStateTransitionOperation* op))block
{
    OctagonStateTransitionOperation* op = [[self alloc] initIntending:intendedState
                                                           errorState:errorState];
    WEAKIFY(op);
    [op addExecutionBlock:^{
        STRONGIFY(op);
        block(op);
    }];
    op.name = name;
    return op;
}

+ (instancetype)named:(NSString*)name
             entering:(OctagonState*)intendedState
{
    OctagonStateTransitionOperation* op = [[self alloc] initIntending:intendedState
                                                           errorState:intendedState];
    op.name = name;
    return op;
}

@end

#pragma mark -- OctagonStateTransitionGroupOperation

@implementation OctagonStateTransitionGroupOperation
- (instancetype)initIntending:(OctagonState*)intendedState
                   errorState:(OctagonState*)errorState
{
    if((self = [super init])) {
        _nextState = errorState;
        _intendedState = intendedState;
    }
    return self;
}

- (NSString*)description
{
    return [NSString stringWithFormat:@"<OctagonStateTransitionGroupOperation(%@): intended:%@ actual:%@>", self.name, self.intendedState, self.nextState];
}

+ (instancetype)named:(NSString*)name
            intending:(OctagonState*)intendedState
           errorState:(OctagonState*)errorState
  withBlockTakingSelf:(void(^)(OctagonStateTransitionGroupOperation* op))block
{
    OctagonStateTransitionGroupOperation* op = [[self alloc] initIntending:intendedState
                                                                errorState:errorState];
    WEAKIFY(op);
    [op runBeforeGroupFinished:[NSBlockOperation blockOperationWithBlock:^{
        STRONGIFY(op);
        block(op);
    }]];
    op.name = name;
    return op;
}
@end

#pragma mark -- OctagonStateTransitionRequest

@interface OctagonStateTransitionRequest ()
@property dispatch_queue_t queue;
@property bool timeoutCanOccur;
@end

@implementation OctagonStateTransitionRequest

- (instancetype)init:(NSString*)name
        sourceStates:(NSSet<OctagonState*>*)sourceStates
         serialQueue:(dispatch_queue_t)queue
        transitionOp:(CKKSResultOperation<OctagonStateTransitionOperationProtocol>*)transitionOp
{
    if((self = [super init])) {
        _name = name;
        _sourceStates = sourceStates;
        _queue = queue;

        _timeoutCanOccur = true;
        _transitionOperation = transitionOp;
    }
    
    return self;
}

- (NSString*)description
{
    return [NSString stringWithFormat:@"<OctagonStateTransitionRequest: %@ %@ sources:%d>", self.name, self.transitionOperation, (unsigned int)[self.sourceStates count]];
}

- (CKKSResultOperation<OctagonStateTransitionOperationProtocol>* _Nullable)_onqueueStart
{
    dispatch_assert_queue(self.queue);

    if(self.timeoutCanOccur) {
        self.timeoutCanOccur = false;
        return self.transitionOperation;
    } else {
        return nil;
    }
}

- (void)onqueueHandleStartTimeout:(NSError*)stateMachineStateError
{
    dispatch_assert_queue(self.queue);

    if(self.timeoutCanOccur) {
        self.timeoutCanOccur = false;

        // The operation will only realize it's finished once added to any operation queue. Fake one up.
        self.transitionOperation.descriptionUnderlyingError = stateMachineStateError;
        self.transitionOperation.descriptionErrorCode = CKKSResultDescriptionErrorPendingMachineRequestStart;
        [self.transitionOperation timeout:0*NSEC_PER_SEC];

        [[[NSOperationQueue alloc] init] addOperation:self.transitionOperation];
    }
}

@end

#endif // OCTAGON
