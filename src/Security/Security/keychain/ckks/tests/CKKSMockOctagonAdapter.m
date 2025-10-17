/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 4, 2022.
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

#import <CloudKit/CloudKit.h>
#import <SecurityFoundation/SFKey.h>
#import <SecurityFoundation/SFKey_Private.h>
#import <SecurityFoundation/SFDigestOperation.h>

#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSKey.h"
#import "keychain/ckks/CKKSViewManager.h"
#import "keychain/ckks/CloudKitCategories.h"
#import "keychain/categories/NSError+UsefulConstructors.h"

#import "keychain/ot/OctagonCKKSPeerAdapter.h"

#import "keychain/ckks/CKKSListenerCollection.h"
#import "keychain/ckks/tests/CKKSMockOctagonAdapter.h"

@implementation CKKSMockOctagonPeer

@synthesize publicSigningKey = _publicSigningKey;
@synthesize publicEncryptionKey = _publicEncryptionKey;


- (instancetype)initWithOctagonPeerID:(NSString*)syncingPeerID
                  publicEncryptionKey:(SFECPublicKey* _Nullable)publicEncryptionKey
                     publicSigningKey:(SFECPublicKey* _Nullable)publicSigningKey
                             viewList:(NSSet<NSString*>* _Nullable)viewList
{
    if((self = [super init])) {
        _peerID = syncingPeerID;
        _publicEncryptionKey = publicEncryptionKey;
        _publicSigningKey = publicSigningKey;
        _viewList = viewList;
    }
    return self;
}

- (bool)matchesPeer:(nonnull id<CKKSPeer>)peer {
    NSString* otherPeerID = peer.peerID;

    if(self.peerID == nil && otherPeerID == nil) {
        return true;
    }

    return [self.peerID isEqualToString:otherPeerID];
}

- (BOOL)shouldHaveView:(nonnull NSString *)viewName {
    return [self.viewList containsObject: viewName];
}

@end


@implementation CKKSMockOctagonAdapter
@synthesize essential = _essential;

- (instancetype)initWithSelfPeer:(OctagonSelfPeer*)selfPeer
                    trustedPeers:(NSSet<id<CKKSRemotePeerProtocol>>*)trustedPeers
                       essential:(BOOL)essential
{
    if((self = [super init])) {
        _essential = essential;

        _peerChangeListeners = [[CKKSListenerCollection alloc] initWithName:@"ckks-mock-sos"];

        _selfOTPeer = selfPeer;
        _trustedPeers = [trustedPeers mutableCopy];
    }
    return self;
}

- (NSString*)providerID
{
    return [NSString stringWithFormat:@"[CKKSMockOctagonAdapter: %@]", self.selfOTPeer.peerID];
}

- (CKKSSelves * _Nullable)fetchSelfPeers:(NSError * _Nullable __autoreleasing * _Nullable)error {
    return [[CKKSSelves alloc] initWithCurrent:self.selfOTPeer allSelves:nil];
}

- (NSSet<id<CKKSRemotePeerProtocol>> * _Nullable)fetchTrustedPeers:(NSError * _Nullable __autoreleasing * _Nullable)error {
    // include the self peer
    CKKSMockOctagonPeer *s = [[CKKSMockOctagonPeer alloc] initWithOctagonPeerID:self.selfOTPeer.peerID
                                                            publicEncryptionKey:self.selfOTPeer.publicEncryptionKey
                                                               publicSigningKey:self.selfOTPeer.publicSigningKey
                                                                       viewList:self.selfViewList];
    return [self.trustedPeers setByAddingObject: s];
}

- (void)registerForPeerChangeUpdates:(nonnull id<CKKSPeerUpdateListener>)listener {
    [self.peerChangeListeners registerListener:listener];
}

- (void)sendSelfPeerChangedUpdate {
    [self.peerChangeListeners iterateListeners: ^(id<CKKSPeerUpdateListener> listener) {
        [listener selfPeerChanged:self];
    }];
}

- (void)sendTrustedPeerSetChangedUpdate {
    [self.peerChangeListeners iterateListeners: ^(id<CKKSPeerUpdateListener> listener) {
        [listener trustedPeerSetChanged:self];
    }];
}

- (nonnull CKKSPeerProviderState *)currentState {
    return [CKKSPeerProviderState createFromProvider:self];
}


@end

#endif
