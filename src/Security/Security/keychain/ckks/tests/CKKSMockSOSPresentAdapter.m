/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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

#import "keychain/ckks/CKKSListenerCollection.h"
#import "keychain/ckks/tests/CKKSMockSOSPresentAdapter.h"

@interface CKKSMockSOSPresentAdapter()
@property CKKSListenerCollection* peerChangeListeners;
@property bool isSOSEnabled;
@end

@implementation CKKSMockSOSPresentAdapter
@synthesize essential = _essential;

- (instancetype)initWithSelfPeer:(CKKSSOSSelfPeer*)selfPeer
                    trustedPeers:(NSSet<id<CKKSSOSPeerProtocol>>*)trustedPeers
                       essential:(BOOL)essential
{
    if((self = [super init])) {
        _essential = essential;
        _isSOSEnabled = true;
        
        _circleStatus = kSOSCCInCircle;
        _safariViewEnabled = YES;

        _excludeSelfPeerFromTrustSet = false;

        _peerChangeListeners = [[CKKSListenerCollection alloc] initWithName:@"ckks-mock-sos"];

        _ckks4AllStatus = NO;
        _ckks4AllStatusIsSet = NO;

        _selfPeer = selfPeer;
        _trustedPeers = [trustedPeers mutableCopy];
        
        _joinAfterRestoreResult = false;
        _resetToOfferingResult = false;

        _joinAfterRestoreCircleStatusOverride = false;
        _resetToOfferingCircleStatusOverride = false;
    }
    return self;
}

- (bool)sosEnabled
{
    return self.isSOSEnabled;
}

- (void)setSOSEnabled:(bool)isEnabled
{
    self.isSOSEnabled = isEnabled;
}

- (NSString*)providerID
{
    return [NSString stringWithFormat:@"[CKKSMockSOSPresentAdapter: %@]", self.selfPeer.peerID];
}

- (SOSCCStatus)circleStatus:(NSError * _Nullable __autoreleasing * _Nullable)error
{
    if(![self isSOSEnabled] || self.circleStatus == kSOSCCError) {
        if(error && self.circleStatus == kSOSCCError) {
            // I'm not at all sure that the second error here actually is any error in particular
            *error = self.circleStatusError ?: [NSError errorWithDomain:(__bridge NSString*)kSOSErrorDomain code:self.circleStatus userInfo:nil];
        }
        return kSOSCCError;
    }

    return self.circleStatus;
}

// I currently don't understand when SOS returns a self or not. I've seen it return a self while not in kSOSCCInCircle,
// which seems wrong. So, always return a self, unless we're in an obvious error state.
- (id<CKKSSelfPeer> _Nullable)currentSOSSelf:(NSError * _Nullable __autoreleasing * _Nullable)error
{
    if(self.selfPeerError) {
        if(error) {
            *error = self.selfPeerError;
        }
        return nil;
    }

    if(self.aksLocked) {
        if(error) {
            *error = [NSError errorWithDomain:NSOSStatusErrorDomain code:errSecInteractionNotAllowed userInfo:nil];
        }
        return nil;
    }

    if([self isSOSEnabled] && self.circleStatus == kSOSCCInCircle) {
        return self.selfPeer;
    } else {
        if(error) {
            *error = [NSError errorWithDomain:(__bridge NSString*)kSOSErrorDomain code:self.circleStatus userInfo:nil];
        }
        return nil;
    }
}

- (CKKSSelves * _Nullable)fetchSelfPeers:(NSError *__autoreleasing  _Nullable * _Nullable)error
{
    id<CKKSSelfPeer> peer = [self currentSOSSelf:error];
    if(!peer) {
        return nil;
    }

    return [[CKKSSelves alloc] initWithCurrent:peer allSelves:nil];
}

- (NSSet<id<CKKSRemotePeerProtocol>> * _Nullable)fetchTrustedPeers:(NSError * _Nullable __autoreleasing * _Nullable)error
{
    if(self.trustedPeersError) {
        if(error) {
            *error = self.trustedPeersError;
        }
        return nil;
    }

    // TODO: I'm actually not entirely sure what SOS does if it's not in circle?
    if([self isSOSEnabled] && self.circleStatus == kSOSCCInCircle) {
        if(self.excludeSelfPeerFromTrustSet) {
            return self.trustedPeers;
        } else {
            return [self allPeers];
        }
    } else {
        if(error) {
            *error = [NSError errorWithDomain:(__bridge NSString*)kSOSErrorDomain code:kSOSCCNotInCircle userInfo:nil];
        }
        return nil;
    }
}

- (BOOL)updateOctagonKeySetWithAccount:(nonnull id<CKKSSelfPeer>)currentSelfPeer error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    if(self.updateOctagonKeySetListener) {
        self.updateOctagonKeySetListener(currentSelfPeer);
    }
    return YES;
}

- (BOOL)updateCKKS4AllStatus:(BOOL)status error:(NSError**)error
{
    self.ckks4AllStatus = status;
    self.ckks4AllStatusIsSet = YES;
    return YES;
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

- (NSSet<id<CKKSRemotePeerProtocol>>*)allPeers
{
    // include the self peer, but as a CKKSSOSPeer object instead of a self peer
    CKKSSOSPeer* s = [[CKKSSOSPeer alloc] initWithSOSPeerID:self.selfPeer.peerID
                                        encryptionPublicKey:self.selfPeer.publicEncryptionKey
                                           signingPublicKey:self.selfPeer.publicSigningKey
                                                   viewList:self.selfPeer.viewList];

    return [self.trustedPeers setByAddingObject: s];
}

- (BOOL)safariViewSyncingEnabled:(NSError**)error
{
    // TODO: what happens if you call this when not in circle?
    return self.safariViewEnabled;
}

- (BOOL)preloadOctagonKeySetOnAccount:(nonnull id<CKKSSelfPeer>)currentSelfPeer error:(NSError *__autoreleasing  _Nullable * _Nullable)error {
    // No-op
    return YES;
}

- (bool)joinAfterRestore:(NSError * _Nullable __autoreleasing * _Nullable)error
{
    if (self.joinAfterRestoreCircleStatusOverride == false) {
        if(!self.joinAfterRestoreResult) {
            self.circleStatus = kSOSCCNotInCircle;
        } else {
            self.circleStatus = kSOSCCInCircle;
        }
    }
    return self.joinAfterRestoreResult;
}

- (bool)resetToOffering:(NSError * _Nullable __autoreleasing * _Nullable)error
{
    if (self.resetToOfferingCircleStatusOverride == false) {
        if(!self.resetToOfferingResult) {
            self.circleStatus = kSOSCCNotInCircle;
        } else {
            self.circleStatus = kSOSCCInCircle;
        }
    }
    
    return self.resetToOfferingResult;
}

@end
