/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
#ifndef OTCustodianRecoveryKey_h
#define OTCustodianRecoveryKey_h

#if __OBJC2__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// All the information related to the custodian recovery key.
@interface OTCustodianRecoveryKey : NSObject <NSSecureCoding>

- (nullable instancetype)initWithUUID:(NSUUID*)uuid recoveryString:(NSString*)recoveryString error:(NSError**)error;
- (nullable instancetype)initWithWrappedKey:(NSData*)wrappedKey wrappingKey:(NSData*)wrappingKey uuid:(NSUUID*)uuid error:(NSError**)error;

- (BOOL)isEqualToCustodianRecoveryKey:(OTCustodianRecoveryKey*)other;
- (BOOL)isEqual:(nullable id)object;

- (NSDictionary*)dictionary;

@property (strong, nonatomic, readonly) NSUUID *uuid;               // Unique identifier for each CRK
@property (strong, nonatomic, readonly) NSData *wrappingKey;        // Key to encrypt recoveryString -- to create two parts both needed to reassemble.
@property (strong, nonatomic, readonly) NSData *wrappedKey;         // The recoveryString encrypted by wrappingKey (IV + ciphertext) -- in AES GCM.
@property (strong, nonatomic, readonly) NSString *recoveryString;   // random string that is used to derive encryptionKey and signingKey.
@end

NS_ASSUME_NONNULL_END

#endif /* OBJC2 */

#endif /* OTCustodianRecoveryKey_h */
