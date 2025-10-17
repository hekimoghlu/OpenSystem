/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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

#import <SecurityFoundation/SecurityFoundation.h>
#import "keychain/ot/OTPreloadOctagonKeysOperation.h"
#import "keychain/ot/OTCuttlefishContext.h"
#import "keychain/ot/OTDefines.h"
#import "keychain/ot/OTConstants.h"
#import "keychain/ot/OctagonCKKSPeerAdapter.h"
#import "utilities/debugging.h"
#import <Security/SecKey.h>
#import <Security/SecKeyPriv.h>

#import "keychain/TrustedPeersHelper/TrustedPeersHelperProtocol.h"
#import "keychain/ot/ObjCImprovements.h"
#import "keychain/securityd/SOSCloudCircleServer.h"

@interface OTPreloadOctagonKeysOperation ()
@property OTOperationDependencies* deps;

@property NSOperation* finishOp;
@end

@implementation OTPreloadOctagonKeysOperation
@synthesize intendedState = _intendedState;

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
    secnotice("octagon-preload-keys", "Beginning operation that preloads the SOSAccount with newly created Octagon Keys");

    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];

    if(!self.deps.sosAdapter.sosEnabled) {
        self.error = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorSOSAdapter userInfo:@{NSLocalizedDescriptionKey : @"sos adapter not enabled"}];
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    NSError* fetchSelfPeersError = nil;
    CKKSSelves *selfPeers = [self.deps.octagonAdapter fetchSelfPeers:&fetchSelfPeersError];
    if((!selfPeers) || fetchSelfPeersError) {
        secnotice("octagon-preload-keys", "failed to retrieve self peers: %@", fetchSelfPeersError);
        self.error = fetchSelfPeersError;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    id<CKKSSelfPeer> currentSelfPeer = selfPeers.currentSelf;
    if(currentSelfPeer == nil) {
        secnotice("octagon-preload-keys", "failed to retrieve current self");
        self.error = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorOctagonAdapter userInfo: @{ NSLocalizedDescriptionKey : @"failed to retrieve current self"}];
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    NSError* updateError = nil;
    BOOL ret = [self.deps.sosAdapter preloadOctagonKeySetOnAccount:currentSelfPeer error:&updateError];
    if(!ret) {
        self.error = updateError;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    self.nextState = self.intendedState;
    [self runBeforeGroupFinished:self.finishOp];
}

@end

#endif // OCTAGON
