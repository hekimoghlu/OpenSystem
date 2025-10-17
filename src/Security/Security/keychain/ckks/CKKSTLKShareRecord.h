/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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

#import "keychain/ckks/CKKS.h"
#import "keychain/ckks/CKKSItem.h"
#import "keychain/ckks/CKKSKey.h"
#import "keychain/ckks/CKKSPeer.h"
#import "keychain/ckks/CKKSTLKShare.h"

#import <SecurityFoundation/SFEncryptionOperation.h>
#import <SecurityFoundation/SFKey.h>

NS_ASSUME_NONNULL_BEGIN

@interface CKKSTLKShareRecord : CKKSCKRecordHolder
@property CKKSTLKShare* share;

// Passthroughs to the underlying share
@property (readonly) NSString* tlkUUID;

@property (readonly) NSString* senderPeerID;

@property (readonly) NSInteger epoch;
@property (readonly) NSInteger poisoned;

@property (readonly, nullable) NSData* wrappedTLK;
@property (readonly, nullable) NSData* signature;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithShare:(CKKSTLKShare*)share
                    contextID:(NSString*)contextID
                       zoneID:(CKRecordZoneID*)zoneID
              encodedCKRecord:(NSData* _Nullable)encodedCKRecord;

- (CKKSKeychainBackedKey* _Nullable)recoverTLK:(id<CKKSSelfPeer>)recoverer
                                  trustedPeers:(NSSet<id<CKKSPeer>>*)peers
                                         error:(NSError**)error;

+ (CKKSTLKShareRecord* _Nullable)share:(CKKSKeychainBackedKey*)key
                             contextID:(NSString*)contextID
                                    as:(id<CKKSSelfPeer>)sender
                                    to:(id<CKKSPeer>)receiver
                                 epoch:(NSInteger)epoch
                              poisoned:(NSInteger)poisoned
                                 error:(NSError**)error;

- (bool)signatureVerifiesWithPeerSet:(NSSet<id<CKKSPeer>>*)peerSet error:(NSError**)error;

- (NSData*)dataForSigning;

// Database loading
+ (instancetype _Nullable)fromDatabase:(NSString*)uuid
                             contextID:(NSString*)contextID
                        receiverPeerID:(NSString*)receiverPeerID
                          senderPeerID:(NSString*)senderPeerID
                                zoneID:(CKRecordZoneID*)zoneID
                                 error:(NSError* __autoreleasing*)error;

+ (instancetype _Nullable)tryFromDatabase:(NSString*)uuid
                                contextID:(NSString*)contextID
                           receiverPeerID:(NSString*)receiverPeerID
                             senderPeerID:(NSString*)senderPeerID
                                   zoneID:(CKRecordZoneID*)zoneID
                                    error:(NSError**)error;

+ (NSArray<CKKSTLKShareRecord*>*)allFor:(NSString*)receiverPeerID
                              contextID:(NSString*)contextID
                                keyUUID:(NSString*)uuid
                                 zoneID:(CKRecordZoneID*)zoneID
                                  error:(NSError* __autoreleasing*)error;

+ (NSArray<CKKSTLKShareRecord*>*)allForUUID:(NSString*)uuid
                                  contextID:(NSString*)contextID
                                     zoneID:(CKRecordZoneID*)zoneID
                                      error:(NSError**)error;

+ (NSArray<CKKSTLKShareRecord*>*)allInZone:(CKRecordZoneID*)zoneID
                                 contextID:(NSString*)contextID
                                     error:(NSError**)error;

+ (instancetype _Nullable)tryFromDatabaseFromCKRecordID:(CKRecordID*)recordID
                                              contextID:(NSString*)contextID
                                                  error:(NSError**)error;

// Returns a prefix that all every CKKSTLKShare CKRecord will have
+ (NSString*)ckrecordPrefix;

+ (BOOL)intransactionRecordChanged:(CKRecord*)record
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error;

+ (BOOL)intransactionRecordDeleted:(CKRecordID*)recordID
                         contextID:(NSString*)contextID
                            resync:(BOOL)resync
                             error:(NSError**)error;

+ (NSNumber* _Nullable)countsWithContextID:(NSString*)contextID
                                    zoneID:(CKRecordZoneID*)zoneID
                                     error:(NSError * __autoreleasing *)error;

// For tests
- (NSData* _Nullable)signRecord:(SFECKeyPair*)signingKey error:(NSError**)error;
- (bool)verifySignature:(NSData*)signature verifyingPeer:(id<CKKSPeer>)peer error:(NSError**)error;
@end

NS_ASSUME_NONNULL_END

#endif  // OCTAGON
