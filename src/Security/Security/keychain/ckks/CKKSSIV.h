/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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

NS_ASSUME_NONNULL_BEGIN

// For AES-SIV 512.

#define CKKSKeySize (512 / 8)
#define CKKSWrappedKeySize (CKKSKeySize + 16)

@interface CKKSBaseAESSIVKey : NSObject <NSCopying>
{
   @package
    uint8_t key[CKKSWrappedKeySize];  // subclasses can use less than the whole buffer, and set key to be precise
    size_t size;
}
- (instancetype)init;
- (instancetype)initWithBytes:(uint8_t*)bytes len:(size_t)len;
- (void)zeroKey;
- (instancetype)copyWithZone:(NSZone* _Nullable)zone;

// Mostly for testing.
- (instancetype)initWithBase64:(NSString*)base64bytes;
- (BOOL)isEqual:(id _Nullable)object;
@end

@interface CKKSWrappedAESSIVKey : CKKSBaseAESSIVKey <NSSecureCoding>
- (instancetype)initWithData:(NSData*)data;
- (NSData*)wrappedData;
- (NSString*)base64WrappedKey;

// Almost certainly not a valid key; use if you need a placeholder
+ (CKKSWrappedAESSIVKey*)zeroedKey;
@end

@interface CKKSAESSIVKey : CKKSBaseAESSIVKey
+ (instancetype _Nullable)randomKey:(NSError*__autoreleasing*)error;

- (CKKSWrappedAESSIVKey* _Nullable)wrapAESKey:(CKKSAESSIVKey*)keyToWrap
                                        error:(NSError* __autoreleasing*)error;

- (CKKSAESSIVKey* _Nullable)unwrapAESKey:(CKKSWrappedAESSIVKey*)keyToUnwrap
                                   error:(NSError* __autoreleasing*)error;

// Encrypt and decrypt data into buffers. Adds a nonce for ciphertext protection.
- (NSData* _Nullable)encryptData:(NSData*)plaintext
               authenticatedData:(NSDictionary<NSString*, NSData*>* _Nullable)ad
                           error:(NSError* __autoreleasing*)error;
- (NSData* _Nullable)decryptData:(NSData*)ciphertext
               authenticatedData:(NSDictionary<NSString*, NSData*>* _Nullable)ad
                           error:(NSError* __autoreleasing*)error;

// Please only call this if you're storing this key to the keychain, or sending it to a peer.
- (NSData*)keyMaterial;

@end

NS_ASSUME_NONNULL_END

#endif  // OCTAGON
