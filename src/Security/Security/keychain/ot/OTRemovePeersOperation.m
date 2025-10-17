/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 8, 2024.
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

#import "keychain/ot/OTRemovePeersOperation.h"
#import "keychain/ot/OTOperationDependencies.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"

@interface OTRemovePeersOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishedOp;
@end

@implementation OTRemovePeersOperation
@synthesize intendedState = _intendedState;
@synthesize nextState = _nextState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                             peerIDs:(NSArray<NSString*>*)peerIDs
{
    if((self = [super init])) {
        _deps = dependencies;

        _intendedState = intendedState;
        _nextState = errorState;

        _peerIDs = [NSSet setWithArray:peerIDs];
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon", "Attempting to remove peers: %@", self.peerIDs);

    self.finishedOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishedOp];

    WEAKIFY(self);
    [self.deps.cuttlefishXPCWrapper distrustPeerIDsWithSpecificUser:self.deps.activeAccount
                                                            peerIDs:self.peerIDs
                                                              reply:^(NSError * _Nullable error) {
        STRONGIFY(self);
        if(error) {
            secnotice("octagon", "Unable to remove peers for (%@,%@): %@", self.deps.containerName, self.deps.contextID, error);
            self.error = error;
        } else {
            secnotice("octagon", "Successfully removed peers");
        }

        [self runBeforeGroupFinished:self.finishedOp];
    }];
}

@end

#endif // OCTAGON

