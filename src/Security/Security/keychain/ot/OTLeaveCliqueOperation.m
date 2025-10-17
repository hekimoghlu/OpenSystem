/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 17, 2021.
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

#import "keychain/ot/OTLeaveCliqueOperation.h"
#import "keychain/ot/OTOperationDependencies.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/OTStates.h"

@interface OTLeaveCliqueOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishedOp;
@end

@implementation OTLeaveCliqueOperation
@synthesize intendedState = _intendedState;
@synthesize nextState = _nextState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
{
    if((self = [super init])) {
        _deps = dependencies;

        _intendedState = intendedState;
        _nextState = errorState;
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon", "Attempting to leave clique");

    self.finishedOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishedOp];

    WEAKIFY(self);
    [self.deps.cuttlefishXPCWrapper departByDistrustingSelfWithSpecificUser:self.deps.activeAccount
                                                                      reply:^(NSError * _Nullable error) {
            STRONGIFY(self);
            if(error) {
                self.error = error;
                if([self.deps.lockStateTracker isLockedError:error]) {
                    secnotice("octagon", "Departing failed due to lock state: %@", error);
                    self.nextState = OctagonStateWaitForUnlock;
                } else {
                    secnotice("octagon", "Unable to depart for (%@,%@): %@", self.deps.containerName, self.deps.contextID, error);
                }
            } else {
                NSError* localError = nil;
                BOOL persisted = [self.deps.stateHolder persistNewTrustState:OTAccountMetadataClassC_TrustState_UNTRUSTED
                                                                       error:&localError];
                if(!persisted || localError) {
                    secerror("octagon: unable to persist clique departure: %@", localError);
                    self.error = localError;
                } else {
                    secnotice("octagon", "Successfully departed clique");
                    self.nextState = self.intendedState;
                }
            }

            [self runBeforeGroupFinished:self.finishedOp];
        }];
}

@end

#endif // OCTAGON
