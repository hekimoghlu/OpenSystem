/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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

#import <utilities/debugging.h>

#import "keychain/ot/OTPreflightVouchWithCustodianRecoveryKeyOperation.h"
#import "keychain/ot/OTStates.h"

#import "keychain/OctagonTrust/OTCustodianRecoveryKey.h"

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"

@interface OTPreflightVouchWithCustodianRecoveryKeyOperation ()
@property OTOperationDependencies* deps;

@property NSString* salt;

@property NSOperation* finishOp;

@property TrustedPeersHelperCustodianRecoveryKey *tphcrk;

@end

@implementation OTPreflightVouchWithCustodianRecoveryKeyOperation
@synthesize intendedState = _intendedState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
                              tphcrk:(TrustedPeersHelperCustodianRecoveryKey*)tphcrk
{
    if((self = [super init])) {
        _deps = dependencies;
        _intendedState = intendedState;
        _nextState = errorState;

        _tphcrk = tphcrk;
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon", "creating voucher using a custodian recovery key");

    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];

    // First, let's preflight the vouch (to receive a policy and view set to use for TLK fetching
    WEAKIFY(self);
    [self.deps.cuttlefishXPCWrapper preflightVouchWithCustodianRecoveryKeyWithSpecificUser:self.deps.activeAccount
                                                                                       crk:self.tphcrk
                                                                                     reply:^(NSString * _Nullable recoveryKeyID,
                                                                                             TPSyncingPolicy* _Nullable peerSyncingPolicy,
                                                                                             NSError * _Nullable error) {
        STRONGIFY(self);
        [[CKKSAnalytics logger] logResultForEvent:OctagonEventPreflightVouchWithCustodianRecoveryKey hardFailure:true result:error];

        if(error || !recoveryKeyID) {
            secerror("octagon: Error preflighting voucher using custodian recovery key: %@", error);
            self.error = error;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }

        secnotice("octagon", "Preflight Custodian Recovery key ID %@ looks good to go", recoveryKeyID);

        self.nextState = self.intendedState;
        [self runBeforeGroupFinished:self.finishOp];
    }];
}

@end

#endif // OCTAGON
