/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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

//
//  CKKSMockOctagonAdapter.h
//  Security
//
//  Created by Love HÃ¶rnquist Ã…strand on 6/19/19.
//

#import <Foundation/Foundation.h>

#import "keychain/ckks/CKKSPeer.h"
#import "keychain/ckks/CKKSListenerCollection.h"
#import "keychain/ot/OctagonCKKSPeerAdapter.h"

NS_ASSUME_NONNULL_BEGIN

@interface CKKSMockOctagonPeer : NSObject <CKKSPeer, CKKSRemotePeerProtocol>

@property NSString* peerID;
@property (nullable) SFECPublicKey* publicEncryptionKey;
@property (nullable) SFECPublicKey* publicSigningKey;
@property (nullable) NSSet<NSString*>* viewList;

- (instancetype)initWithOctagonPeerID:(NSString*)syncingPeerID
                  publicEncryptionKey:(SFECPublicKey* _Nullable)publicEncryptionKey
                     publicSigningKey:(SFECPublicKey* _Nullable)publicSigningKey
                             viewList:(NSSet<NSString*>* _Nullable)viewList;
@end

@interface CKKSMockOctagonAdapter : NSObject <CKKSPeerProvider>
@property CKKSListenerCollection* peerChangeListeners;
@property OctagonSelfPeer* selfOTPeer;
@property (nullable) NSSet<NSString*>* selfViewList;

@property NSMutableSet<id<CKKSRemotePeerProtocol>>* trustedPeers;
@property (readonly) NSString* providerID;

- (instancetype)initWithSelfPeer:(OctagonSelfPeer*)selfPeer
                    trustedPeers:(NSSet<id<CKKSRemotePeerProtocol>>*)trustedPeers
                       essential:(BOOL)essential;

@end


NS_ASSUME_NONNULL_END
