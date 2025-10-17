/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
#import "keychain/ckks/CKKSRecordHolder.h"
#import "keychain/ckks/CKKSSIV.h"
#import "keychain/ckks/proto/generated_source/CKKSSerializedKey.h"

NS_ASSUME_NONNULL_BEGIN

// Important note: while this class does conform to NSSecureCoding,
// for safety reasons encoding a CKKSKeychainBackedKey will ~not~
// encode the aessivkey. If you want your receiver to have access
// to the original key material, they must successfully call
// loadKeyMaterialFromKeychain.

@interface CKKSKeychainBackedKey : NSObject <NSCopying, NSSecureCoding>
@property NSString* uuid;
@property NSString* parentKeyUUID;
@property CKKSKeyClass* keyclass;
@property CKRecordZoneID* zoneID;

// Actual key material
@property CKKSWrappedAESSIVKey* wrappedkey;
@property (nullable) CKKSAESSIVKey* aessivkey;

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithWrappedAESKey:(CKKSWrappedAESSIVKey* _Nullable)wrappedaeskey
                                 uuid:(NSString*)uuid
                        parentKeyUUID:(NSString*)parentKeyUUID
                             keyclass:(CKKSKeyClass*)keyclass
                               zoneID:(CKRecordZoneID*)zoneID;

// Creates new random keys, in the parent's zone
+ (instancetype _Nullable)randomKeyWrappedByParent:(CKKSKeychainBackedKey*)parentKey
                                             error:(NSError* __autoreleasing*)error;

+ (instancetype _Nullable)randomKeyWrappedByParent:(CKKSKeychainBackedKey*)parentKey
                                          keyclass:(CKKSKeyClass*)keyclass
                                             error:(NSError* __autoreleasing*)error;

// Creates a new random key that wraps itself
+ (instancetype _Nullable)randomKeyWrappedBySelf:(CKRecordZoneID*)zoneID
                                           error:(NSError* __autoreleasing*)error;

+ (instancetype _Nullable)keyWrappedBySelf:(CKKSAESSIVKey*)aeskey
                                      uuid:(NSString*)uuid
                                  keyclass:(CKKSKeyClass*)keyclass
                                    zoneID:(CKRecordZoneID*)zoneID
                                     error:(NSError**)error;

/* Helper functions for persisting key material in the keychain */
- (BOOL)saveKeyMaterialToKeychain:(NSError* __autoreleasing*)error;
- (BOOL)saveKeyMaterialToKeychain:(bool)stashTLK
                            error:(NSError* __autoreleasing*)error;  // call this to not stash a non-syncable TLK, if that's what you want

- (BOOL)loadKeyMaterialFromKeychain:(NSError* __autoreleasing*)error;
- (BOOL)deleteKeyMaterialFromKeychain:(NSError* __autoreleasing*)error;

// Class methods to help tests
+ (NSDictionary* _Nullable)setKeyMaterialInKeychain:(NSDictionary*)query
                                              error:(NSError* __autoreleasing*)error;

+ (NSDictionary* _Nullable)queryKeyMaterialInKeychain:(NSDictionary*)query
                                                error:(NSError* __autoreleasing*)error;

/* Returns true if we believe this key wraps itself. */
- (bool)wrapsSelf;

// Attempts checks if the AES key is already loaded, or attempts to load it from the keychain. Returns nil if it fails.
- (CKKSAESSIVKey* _Nullable)ensureKeyLoadedFromKeychain:(NSError* __autoreleasing*)error;

// On a self-wrapped key, determine if this AES-SIV key is the self-wrapped key.
// If it is, save the key as this CKKSKey's unwrapped key.
- (bool)trySelfWrappedKeyCandidate:(CKKSAESSIVKey*)candidate
                             error:(NSError* __autoreleasing*)error;

- (CKKSWrappedAESSIVKey* _Nullable)wrapAESKey:(CKKSAESSIVKey*)keyToWrap
                                        error:(NSError* __autoreleasing*)error;

- (CKKSAESSIVKey* _Nullable)unwrapAESKey:(CKKSWrappedAESSIVKey*)keyToUnwrap
                                   error:(NSError* __autoreleasing*)error;

- (bool)wrapUnder:(CKKSKeychainBackedKey*)wrappingKey
            error:(NSError* __autoreleasing*)error;

- (bool)unwrapSelfWithAESKey:(CKKSAESSIVKey*)unwrappingKey
                       error:(NSError* __autoreleasing*)error;

- (NSData* _Nullable)encryptData:(NSData*)plaintext
               authenticatedData:(NSDictionary<NSString*, NSData*>* _Nullable)ad
                           error:(NSError* __autoreleasing*)error;

- (NSData* _Nullable)decryptData:(NSData*)ciphertext
               authenticatedData:(NSDictionary<NSString*, NSData*>* _Nullable)ad
                           error:(NSError* __autoreleasing*)error;

- (NSData* _Nullable)serializeAsProtobuf:(NSError* __autoreleasing*)error;

+ (CKKSKeychainBackedKey* _Nullable)loadFromProtobuf:(NSData*)data
                                               error:(NSError* __autoreleasing*)error;
@end

// Useful when sending keys across interface boundaries
@interface CKKSKeychainBackedKeySet : NSObject <NSSecureCoding>
@property CKKSKeychainBackedKey* tlk;
@property CKKSKeychainBackedKey* classA;
@property CKKSKeychainBackedKey* classC;
@property BOOL newUpload;

- (instancetype)initWithTLK:(CKKSKeychainBackedKey*)tlk
                     classA:(CKKSKeychainBackedKey*)classA
                     classC:(CKKSKeychainBackedKey*)classC
                  newUpload:(BOOL)newUpload;
@end


NS_ASSUME_NONNULL_END

#endif
