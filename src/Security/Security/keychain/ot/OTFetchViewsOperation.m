/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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

#import "keychain/ot/OTFetchViewsOperation.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/categories/OTAccountMetadataClassC+KeychainSupport.h"
#import "keychain/ckks/CKKSAnalytics.h"
#import "keychain/ckks/CKKSKeychainView.h"

@interface OTFetchViewsOperation ()
@property OTOperationDependencies* deps;
@end

@implementation OTFetchViewsOperation
@synthesize intendedState = _intendedState;
@synthesize nextState = _nextState;

- (instancetype)initWithDependencies:(OTOperationDependencies*)dependencies
                       intendedState:(OctagonState*)intendedState
                          errorState:(OctagonState*)errorState
{
    if ((self = [super init])) {
        _deps = dependencies;

        _intendedState = intendedState;
        _nextState = errorState;
    }
    return self;
}

- (void)groupStart
{
    secnotice("octagon", "fetching views");

    //double check the account metadata
    NSError* localError = nil;
    OTAccountMetadataClassC* currentAccountMetadata = [self.deps.stateHolder loadOrCreateAccountMetadata:&localError];

    if (!currentAccountMetadata || localError) {
        secnotice("octagon-ckks", "Failed to load account metadata: %@", localError);
    } else {
        self.isInheritedAccount = currentAccountMetadata.isInheritedAccount;
    }

    WEAKIFY(self);
    [self.deps.cuttlefishXPCWrapper fetchCurrentPolicyWithSpecificUser:self.deps.activeAccount
                                                       modelIDOverride:nil
                                                    isInheritedAccount:self.isInheritedAccount
                                                                 reply:^(TPSyncingPolicy* _Nullable syncingPolicy,
                                                                         TPPBPeerStableInfoUserControllableViewStatus userControllableViewStatusOfPeers,
                                                                         NSError* _Nullable error) {
        STRONGIFY(self);
        [[CKKSAnalytics logger] logResultForEvent:OctagonEventFetchViews hardFailure:true result:error];

        if (error) {
            secerror("octagon: failed to retrieve policy+views: %@", error);
            self.error = error;
            return;
        }

        secnotice("octagon-ckks", "Received syncing policy %@ with view list: %@", syncingPolicy, syncingPolicy.viewList);
        // Write them down before continuing

        NSError* stateError = nil;
        [self.deps.stateHolder persistAccountChanges:^OTAccountMetadataClassC * _Nullable(OTAccountMetadataClassC * _Nonnull metadata) {
            [metadata setTPSyncingPolicy:syncingPolicy];
            return metadata;
        } error:&stateError];

        if(stateError) {
            secerror("octagon: failed to save policy+views: %@", stateError);
            self.error = stateError;
            return;
        }

        [self.deps.ckks setCurrentSyncingPolicy:syncingPolicy];

        self.nextState = self.intendedState;
    }];
}

@end

#endif // OCTAGON
