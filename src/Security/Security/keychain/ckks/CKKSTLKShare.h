/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#import <SecurityFoundation/SFEncryptionOperation.h>
#import <SecurityFoundation/SFKey.h>

#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSKeychainBackedKey.h"
#import "keychain/ckks/CKKSPeer.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, SecCKKSTLKShareVersion) {
    SecCKKSTLKShareVersion0 = 0,  // Signature is over all fields except (signature) and (receiverPublicKey)
    // Unknown fields in the CKRecord will be appended to the end, in sorted order based on column ID

    SecCKKSTLKShareVersion1 = 1,  // TLKShare is created for and by SE peers. We won't be able to process it.
};

#define SecCKKSTLKShareCurrentVersion SecCKKSTLKShareVersion0

// Note that a CKKSTLKShare attempts to be forward-compatible with newly-signed fields
// To use this functionality, pass in a CKRecord to its interfaces. If it has extra data,
// that data will be signed or its signature verified.

@interface CKKSTLKShare : NSObject <NSCopying, NSSecureCoding>

@property SFEllipticCurve curve;
@property SecCKKSTLKShareVersion version;

@property NSString* tlkUUID;

@property NSString* receiverPeerID;
@property NSData* receiverPublicEncryptionKeySPKI;

@property NSString* senderPeerID;

@property NSInteger epoch;
@property NSInteger poisoned;

@property (nullable) NSData* wrappedTLK;
@property (nullable) NSData* signature;

@property CKRecordZoneID* zoneID;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)init:(CKKSKeychainBackedKey*)key
              sender:(id<CKKSSelfPeer>)sender
            receiver:(id<CKKSPeer>)receiver
               curve:(SFEllipticCurve)curve
             version:(SecCKKSTLKShareVersion)version
               epoch:(NSInteger)epoch
            poisoned:(NSInteger)poisoned
              zoneID:(CKRecordZoneID*)zoneID;
- (instancetype)initForKey:(NSString*)tlkUUID
              senderPeerID:(NSString*)senderPeerID
            recieverPeerID:(NSString*)receiverPeerID
  receiverEncPublicKeySPKI:(NSData* _Nullable)publicKeySPKI
                     curve:(SFEllipticCurve)curve
                   version:(SecCKKSTLKShareVersion)version
                     epoch:(NSInteger)epoch
                  poisoned:(NSInteger)poisoned
                wrappedKey:(NSData*)wrappedKey
                 signature:(NSData*)signature
                    zoneID:(CKRecordZoneID*)zoneID;

- (CKKSKeychainBackedKey* _Nullable)recoverTLK:(id<CKKSSelfPeer>)recoverer
                                  trustedPeers:(NSSet<id<CKKSPeer>>*)peers
                                      ckrecord:(CKRecord* _Nullable)ckrecord
                                         error:(NSError* __autoreleasing*)error;

+ (CKKSTLKShare* _Nullable)share:(CKKSKeychainBackedKey*)key
                              as:(id<CKKSSelfPeer>)sender
                              to:(id<CKKSPeer>)receiver
                           epoch:(NSInteger)epoch
                        poisoned:(NSInteger)poisoned
                           error:(NSError**)error;

- (bool)signatureVerifiesWithPeerSet:(NSSet<id<CKKSPeer>>*)peerSet
                            ckrecord:(CKRecord* _Nullable)ckrecord
                               error:(NSError**)error;

// For tests
- (CKKSKeychainBackedKey* _Nullable)unwrapUsing:(id<CKKSSelfPeer>)localPeer
                                          error:(NSError**)error;

- (NSData* _Nullable)signRecord:(SFECKeyPair*)signingKey
                       ckrecord:(CKRecord* _Nullable)ckrecord
                          error:(NSError**)error;

- (bool)verifySignature:(NSData*)signature
          verifyingPeer:(id<CKKSPeer>)peer
               ckrecord:(CKRecord* _Nullable)ckrecord
                  error:(NSError**)error;

// Pass in a CKRecord for forward-compatible signatures
- (NSData*)dataForSigning:(CKRecord* _Nullable)record;
@end

NS_ASSUME_NONNULL_END

#endif  // OCTAGON
