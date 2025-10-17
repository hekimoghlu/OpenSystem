/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 24, 2023.
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
/*!
    @header SecItemFetchOutOfBandPriv
    SecItemFetchOutOfBandPriv defines private Objective-C types and SPI functions for fetching PCS items, bypassing the state machine.
*/

#ifndef _SECURITY_SECITEMFETCHOUTOFBANDPRIV_H_
#define _SECURITY_SECITEMFETCHOUTOFBANDPRIV_H_

#ifdef __OBJC__

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface CKKSCurrentItemQuery : NSObject <NSSecureCoding>
@property (nullable, strong) NSString* identifier;
@property (nullable, strong) NSString* accessGroup;
@property (nullable, strong) NSString* zoneID;

- (instancetype)initWithIdentifier:(NSString*)identifier accessGroup:(NSString*)accessGroup zoneID:(NSString*)zoneID;
@end

@interface CKKSCurrentItemQueryResult : NSObject <NSSecureCoding>
@property (nullable, strong) NSString* identifier;
@property (nullable, strong) NSString* accessGroup;
@property (nullable, strong) NSString* zoneID;
@property (nullable, strong) NSDictionary* decryptedRecord;

- (instancetype)initWithIdentifier:(NSString*)identifier accessGroup:(NSString*)accessGroup zoneID:(NSString*)zoneID decryptedRecord:(NSDictionary*)decryptedRecord;
@end

@interface CKKSPCSIdentityQuery : NSObject <NSSecureCoding>
@property (nullable, strong) NSNumber* serviceNumber;
@property (nullable, strong) NSString* accessGroup;
@property (nullable, strong) NSString* publicKey; // public key as a base-64 encoded string
@property (nullable, strong) NSString* zoneID;

- (instancetype)initWithServiceNumber:(NSNumber*)serviceNumber accessGroup:(NSString*)accessGroup publicKey:(NSString*)publicKey zoneID:(NSString*)zoneID;
@end

@interface CKKSPCSIdentityQueryResult : NSObject <NSSecureCoding>
@property (nullable, strong) NSNumber* serviceNumber;
@property (nullable, strong) NSString* publicKey; // public key as a base-64 encoded string
@property (nullable, strong) NSString* zoneID;
@property (nullable, strong) NSDictionary* decryptedRecord;

- (instancetype)initWithServiceNumber:(NSNumber*)serviceNumber publicKey:(NSString*)publicKey zoneID:(NSString*)zoneID decryptedRecord:(NSDictionary*)decryptedRecord;
@end

/*!
     @function secItemFetchCurrentItemOutOfBand
     @abstract Fetches, for the given array of CKKSCurrentItemQuery, the keychain items that are 'current' across this iCloud account from iCloud itself.
 @param currentItemQueries Array of CKKSCurrentItemQuery. Allows for querying multiple items at a time.
 @param forceFetch bool indicating whether results are force-fetched from CKDB
 @param complete Called to return values: Array of CKKSCurrentItemQueryResult which containes items decrypted and returned as a dictionary, accessed using the 'decryptedRecord' property if such items exist. Otherwise, error.
 */

void SecItemFetchCurrentItemOutOfBand(NSArray<CKKSCurrentItemQuery*>* currentItemQueries, bool forceFetch, void (^complete)(NSArray<CKKSCurrentItemQueryResult*>* currentItems, NSError* error));

/*!
     @function secItemFetchPCSIdentityOutOfBand
     @abstract Fetches, for the given array of CKKSPCSIdentityRequest, the item record referring to the PCS Identity in this iCloud account associated with the PCS service number and PCS public key from iCloud itself.
 @param pcsIdentityQueries Array of CKKSPCSIdentityRequest. Allows for querying multiple items at a time.
 @param forceFetch bool indicating whether results are force-fetched from CKDB
 @param complete Called to return values: Array of CKKSCurrentItemQueryResult which containes items decrypted and returned as a dictionary, accessed using the 'decryptedRecord' property if such items exist. Otherwise, error.
 */
void SecItemFetchPCSIdentityOutOfBand(NSArray<CKKSPCSIdentityQuery*>* pcsIdentityQueries, bool forceFetch, void (^complete)(NSArray<CKKSPCSIdentityQueryResult*>* pcsIdentities, NSError* error));

NS_ASSUME_NONNULL_END

#endif /* __OBJC__ */

#endif /* !_SECURITY_SECITEMFETCHOUTOFBANDPRIV_H_ */
