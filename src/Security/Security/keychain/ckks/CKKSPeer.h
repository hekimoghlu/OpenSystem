/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#import <SecurityFoundation/SFKey.h>
#import <SecurityFoundation/SFKey_Private.h>

NS_ASSUME_NONNULL_BEGIN

// ==== Peer protocols ====

@protocol CKKSPeer <NSObject>
@property (readonly) NSString* peerID;
@property (nullable, readonly) SFECPublicKey* publicEncryptionKey;
@property (nullable, readonly) SFECPublicKey* publicSigningKey;

// Not exactly isEqual, since this only compares peerID
- (bool)matchesPeer:(id<CKKSPeer>)peer;
@end

@protocol CKKSRemotePeerProtocol <CKKSPeer>
- (BOOL)shouldHaveView:(NSString*)viewName;
@end

@protocol CKKSSelfPeer <CKKSPeer>
@property (readonly) SFECKeyPair* encryptionKey;
@property (readonly) SFECKeyPair* signingKey;
@end

// ==== Peer collection protocols ====

@interface CKKSSelves : NSObject
@property id<CKKSSelfPeer> currentSelf;
@property (nullable) NSSet<id<CKKSSelfPeer>>* allSelves;
- (instancetype)initWithCurrent:(id<CKKSSelfPeer>)selfPeer allSelves:(NSSet<id<CKKSSelfPeer>>* _Nullable)allSelves;
@end

extern NSString* const CKKSSOSPeerPrefix;

@interface CKKSActualPeer : NSObject <CKKSPeer, CKKSRemotePeerProtocol, NSSecureCoding>
@property (readonly) NSString* peerID;
@property (nullable, readonly) SFECPublicKey* publicEncryptionKey;
@property (nullable, readonly) SFECPublicKey* publicSigningKey;
@property (nullable, readonly) NSSet<NSString*>* viewList;

- (instancetype)initWithPeerID:(NSString*)syncingPeerID
           encryptionPublicKey:(SFECPublicKey* _Nullable)encryptionKey
              signingPublicKey:(SFECPublicKey* _Nullable)signingKey
                      viewList:(NSSet<NSString*>* _Nullable)viewList;
@end

@protocol CKKSSOSPeerProtocol <NSObject, CKKSRemotePeerProtocol>
@end

@interface CKKSSOSPeer : NSObject <CKKSPeer, CKKSSOSPeerProtocol, CKKSRemotePeerProtocol, NSSecureCoding>
- (instancetype)initWithSOSPeerID:(NSString*)syncingPeerID
              encryptionPublicKey:(SFECPublicKey* _Nullable)encryptionKey
                 signingPublicKey:(SFECPublicKey* _Nullable)signingKey
                         viewList:(NSSet<NSString*>* _Nullable)viewList;
@end

@interface CKKSSOSSelfPeer : NSObject <CKKSPeer, CKKSSOSPeerProtocol, CKKSRemotePeerProtocol, CKKSSelfPeer>
@property (readonly) NSString* peerID;
@property (nullable, readonly) NSSet<NSString*>* viewList;

@property (readonly) SFECPublicKey* publicEncryptionKey;
@property (readonly) SFECPublicKey* publicSigningKey;

@property SFECKeyPair* encryptionKey;
@property SFECKeyPair* signingKey;

- (instancetype)initWithSOSPeerID:(NSString*)syncingPeerID
                    encryptionKey:(SFECKeyPair*)encryptionKey
                       signingKey:(SFECKeyPair*)signingKey
                         viewList:(NSSet<NSString*>* _Nullable)viewList;
@end

NSSet<Class>* CKKSPeerClasses(void);

NS_ASSUME_NONNULL_END
#endif  // OCTAGON
