/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#import "keychain/ot/OTEnsureOctagonKeyConsistency.h"
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

@interface OTEnsureOctagonKeyConsistency ()
@property OTOperationDependencies* deps;

@property NSOperation* finishOp;
@end

@implementation OTEnsureOctagonKeyConsistency
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
    secnotice("octagon-sos", "Beginning ensuring Octagon keys are set properly in SOS");

    self.finishOp = [[NSOperation alloc] init];
    [self dependOnBeforeGroupFinished:self.finishOp];

    if(!self.deps.sosAdapter.sosEnabled) {
        self.error = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorSOSAdapter userInfo:@{NSLocalizedDescriptionKey : @"sos adapter not enabled"}];
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }
    NSError* sosSelfFetchError = nil;
    id<CKKSSelfPeer> sosSelf = [self.deps.sosAdapter currentSOSSelf:&sosSelfFetchError];

    if(!sosSelf || sosSelfFetchError) {
        secnotice("octagon-sos", "Failed to get the current SOS self: %@", sosSelfFetchError);
        self.error = sosSelfFetchError;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    secnotice("octagon", "Fetched SOS Self! Fetching Octagon Adapter now.");

    NSError* getEgoPeerError = nil;
    NSString* octagonPeerID = [self.deps.stateHolder getEgoPeerID:&getEgoPeerError];
    if(getEgoPeerError) {
        secnotice("octagon", "failed to get peer id: %@", getEgoPeerError);
        self.error = getEgoPeerError;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    OctagonCKKSPeerAdapter* octagonAdapter = [[OctagonCKKSPeerAdapter alloc] initWithPeerID:octagonPeerID
                                                                               specificUser:self.deps.activeAccount
                                                                             personaAdapter:self.deps.personaAdapter
                                                                              cuttlefishXPC:self.deps.cuttlefishXPCWrapper];

    secnotice("octagon", "Fetched SOS Self! Fetching Octagon Adapter now: %@", octagonAdapter);

    NSError* fetchSelfPeersError = nil;
    CKKSSelves *selfPeers = [octagonAdapter fetchSelfPeers:&fetchSelfPeersError];

    if((!selfPeers) || fetchSelfPeersError) {
        secnotice("octagon", "failed to retrieve self peers: %@", fetchSelfPeersError);
        self.error = fetchSelfPeersError;
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    id<CKKSSelfPeer> currentSelfPeer = selfPeers.currentSelf;
    if(currentSelfPeer == nil) {
        secnotice("octagon", "failed to retrieve current self");
        self.error = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorOctagonAdapter userInfo: @{ NSLocalizedDescriptionKey : @"failed to retrieve current self"}];
        [self runBeforeGroupFinished:self.finishOp];
        return;
    }

    NSData* octagonSigningKeyData = currentSelfPeer.publicSigningKey.keyData;
    NSData* octagonEncryptionKeyData = currentSelfPeer.publicEncryptionKey.keyData;
    NSData* sosSigningKeyData = sosSelf.publicSigningKey.keyData;
    NSData* sosEncryptionKeyData = sosSelf.publicEncryptionKey.keyData;

    if(![octagonSigningKeyData isEqualToData:sosSigningKeyData] || ![octagonEncryptionKeyData isEqualToData:sosEncryptionKeyData]) {
        secnotice("octagon", "SOS and Octagon signing keys do NOT match! updating SOS");
        NSError* updateError = nil;
        BOOL ret = [self.deps.sosAdapter updateOctagonKeySetWithAccount:currentSelfPeer error:&updateError];
        if(!ret) {
            self.error = updateError;
            [self runBeforeGroupFinished:self.finishOp];
            return;
        }
    } else {
        secnotice("octagon", "SOS and Octagon keys match!");
    }
    self.nextState = self.intendedState;
    [self runBeforeGroupFinished:self.finishOp];
}

@end

#endif // OCTAGON
