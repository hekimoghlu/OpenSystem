/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 15, 2023.
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
#import "keychain/ckks/CKKSAnalytics.h"
#import "keychain/ot/OctagonStateMachineHelpers.h"

NS_ASSUME_NONNULL_BEGIN

@class OctagonStateMachine;

@interface OctagonStateTransitionPathStep : NSObject
@property BOOL successState;
@property (readonly) NSDictionary<OctagonState*, OctagonStateTransitionPathStep*>* followStates;

- (instancetype)initAsSuccess;
- (instancetype)initWithPath:(NSDictionary<OctagonState*, OctagonStateTransitionPathStep*>*)followStates;

- (BOOL)successState;

+ (OctagonStateTransitionPathStep*)success;

// Dict should be a map of states to either:
//  1. A dictionary matching this specifiction
//  2. an OctagonStateTransitionPathStep object (which is likely a success object, but doesn't have to be)
// Any other object will be ignored. A malformed dictionary will be converted into an empty success path.
+ (OctagonStateTransitionPathStep*)pathFromDictionary:(NSDictionary<OctagonState*, id>*)pathDict;
@end


@interface OctagonStateTransitionPath : NSObject
@property OctagonState* initialState;
@property OctagonStateTransitionPathStep* pathStep;

- (instancetype)initWithState:(OctagonState*)initialState
                     pathStep:(OctagonStateTransitionPathStep*)pathSteps;

- (OctagonStateTransitionPathStep*)asPathStep;

// Uses the same rules as OctagonStateTransitionPathStep pathFromDictionary, but selects one of the top-level dictionary keys
// to be the path initialization state. Not well defined if you pass in two keys in the top-level dictionary.
// If the dictionary has no keys in it, returns nil.
+ (OctagonStateTransitionPath* _Nullable)pathFromDictionary:(NSDictionary<OctagonState*, id>*)pathDict;

@end



@protocol OctagonStateTransitionWatcherProtocol
@property (readonly) CKKSResultOperation* result;
- (void)onqueueHandleTransition:(CKKSResultOperation<OctagonStateTransitionOperationProtocol>*)attempt;
- (void)onqueueHandleStartTimeout:(NSError*)stateMachineStateError;
@end

@interface OctagonStateTransitionWatcher : NSObject <OctagonStateTransitionWatcherProtocol>
@property (readonly) NSString* name;
@property (readonly) CKKSResultOperation* result;
@property (readonly) OctagonStateTransitionPath* intendedPath;

// If the initial request times out, the watcher will fail as well.
- (instancetype)initNamed:(NSString*)name
             stateMachine:(OctagonStateMachine*)stateMachine
                     path:(OctagonStateTransitionPath*)path
           initialRequest:(OctagonStateTransitionRequest* _Nullable)initialRequest;
@end

// Reports on if any of the given states are entered
@interface OctagonStateMultiStateArrivalWatcher : NSObject <OctagonStateTransitionWatcherProtocol>
@property (readonly) NSString* name;
@property (readonly) CKKSResultOperation* result;
@property (readonly) NSSet<OctagonState*>* states;
@property (readonly) NSDictionary<OctagonState*, NSError*>* failStates;

- (instancetype)initNamed:(NSString*)name
              serialQueue:(dispatch_queue_t)queue
                   states:(NSSet<OctagonState*>*)states;

- (instancetype)initNamed:(NSString*)name
              serialQueue:(dispatch_queue_t)queue
                   states:(NSSet<OctagonState*>*)states
               failStates:(NSDictionary<OctagonState*, NSError*>*)failStates;

// Called by the state machine if it's already in a state at registration time
- (void)onqueueEnterState:(OctagonState*)state;

// If the watcher is still waiting to complete or timeout, cause it to finish with this error
- (void)completeWithErrorIfPending:(NSError*)error;
@end

NS_ASSUME_NONNULL_END

#endif // OCTAGON
