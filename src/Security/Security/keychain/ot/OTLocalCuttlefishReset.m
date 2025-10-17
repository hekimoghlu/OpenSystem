/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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

#import "keychain/ot/OTLocalCuttlefishReset.h"

#import "keychain/ot/ObjCImprovements.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "utilities/debugging.h"

@interface OTLocalResetOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishedOp;
@end

@implementation OTLocalResetOperation
@synthesize intendedState = _intendedState;
@synthesize nextState = _nextState;

- (instancetype)initWithDependencies:(OTOperationDependencies *)deps
                       intendedState:(OctagonState *)intendedState
                          errorState:(OctagonState *)errorState
{
    if((self = [super init])) {
        _intendedState = intendedState;
        _nextState = errorState;

        _deps = deps;
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon-local-reset", "Resetting local cuttlefish");

    self.finishedOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishedOp];

    WEAKIFY(self);
    [self.deps.cuttlefishXPCWrapper localResetWithSpecificUser:self.deps.activeAccount
                                                         reply:^(NSError * _Nullable error) {
            STRONGIFY(self);
            if(error) {
                secnotice("octagon", "Unable to reset local cuttlefish for (%@,%@): %@", self.deps.containerName, self.deps.contextID, error);
                self.error = error;
            } else {
                secnotice("octagon", "Successfully reset local cuttlefish");

                NSError* localError = nil;
                [self.deps.stateHolder persistAccountChanges:^OTAccountMetadataClassC * _Nonnull(OTAccountMetadataClassC * _Nonnull metadata) {
                    metadata.trustState = OTAccountMetadataClassC_TrustState_UNKNOWN;
                    metadata.peerID = nil;
                    metadata.syncingPolicy = nil;

                    // Don't touch the CDP or account states; those can carry over

                    metadata.voucher = nil;
                    metadata.voucherSignature = nil;
                    metadata.tlkSharesForVouchedIdentitys = nil;
                    metadata.isInheritedAccount = NO;
                    metadata.warmedEscrowCache = NO;
                    metadata.warnedTooManyPeers = NO;

                    return metadata;
                } error:&localError];

                if(localError) {
                    secnotice("octagon", "Error resetting local account state: %@", localError);

                } else {
                    secnotice("octagon", "Successfully reset local account state");
                    self.nextState = self.intendedState;
                }
            }

            [self runBeforeGroupFinished:self.finishedOp];
        }];
}

@end

#endif // OCTAGON
