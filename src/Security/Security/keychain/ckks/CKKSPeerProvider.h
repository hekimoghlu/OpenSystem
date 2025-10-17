/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 29, 2024.
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
#import "keychain/ckks/CKKSPeer.h"
#import "keychain/ckks/CKKSCurrentKeyPointer.h"

NS_ASSUME_NONNULL_BEGIN

@protocol CKKSPeerUpdateListener;
@class CKKSPeerProviderState;

#pragma mark - CKKSPeerProvider protocol
@protocol CKKSPeerProvider <NSObject>
@property (readonly) NSString* providerID;
@property BOOL essential;

- (CKKSSelves* _Nullable)fetchSelfPeers:(NSError* _Nullable __autoreleasing* _Nullable)error;
- (NSSet<id<CKKSRemotePeerProtocol>>* _Nullable)fetchTrustedPeers:(NSError* _Nullable __autoreleasing* _Nullable)error;
// Trusted peers should include self peers

- (void)registerForPeerChangeUpdates:(id<CKKSPeerUpdateListener>)listener;
- (void)sendSelfPeerChangedUpdate;
- (void)sendTrustedPeerSetChangedUpdate;

- (CKKSPeerProviderState*)currentState;
@end

#pragma mark - CKKSPeerUpdateListener protocol
// A CKKSPeerUpdateListener wants to be notified when a CKKSPeerProvider has new information
@protocol CKKSPeerUpdateListener <NSObject>
- (void)selfPeerChanged:(id<CKKSPeerProvider> _Nullable)provider;
- (void)trustedPeerSetChanged:(id<CKKSPeerProvider> _Nullable)provider;
@end


#pragma mark - CKKSPeerProviderState

@class CKKSKey;
@class CKKSTLKShareRecord;

@interface CKKSPeerProviderState : NSObject
@property NSString* peerProviderID;

// The peer provider believes trust in this state is essential. Any subsystem using
// a peer provider state should fail and pause if this is YES and there are trust errors.
@property BOOL essential;

@property (nonatomic, readonly, nullable) CKKSSelves* currentSelfPeers;
@property (nonatomic, readonly, nullable) NSError* currentSelfPeersError;
@property (nonatomic, readonly, nullable) NSSet<id<CKKSRemotePeerProtocol>>* currentTrustedPeers;
@property (nonatomic, readonly, nullable) NSSet<NSString*>* currentTrustedPeerIDs;
@property (nonatomic, readonly, nullable) NSError* currentTrustedPeersError;

- (instancetype)initWithPeerProviderID:(NSString*)providerID
                             essential:(BOOL)essential
                             selfPeers:(CKKSSelves* _Nullable)selfPeers
                        selfPeersError:(NSError* _Nullable)selfPeersError
                          trustedPeers:(NSSet<id<CKKSRemotePeerProtocol>>* _Nullable)currentTrustedPeers
                     trustedPeersError:(NSError* _Nullable)trustedPeersError;

- (BOOL)unwrapKey:(CKKSKey*)proposedTLK
       fromShares:(NSArray<CKKSTLKShareRecord*>*)tlkShares
            error:(NSError**)error;

+ (CKKSPeerProviderState*)noPeersState:(id<CKKSPeerProvider>)provider;

// Intended for use in PeerProviders. Thread-safety is up to the PeerProvider.
+ (CKKSPeerProviderState*)createFromProvider:(id<CKKSPeerProvider>)provider;
@end




NS_ASSUME_NONNULL_END


#endif
