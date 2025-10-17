/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 2, 2024.
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
/*
 * SecItemFetchOutOfBandPriv.m - Implements private Objective-C types and SPI functions for fetching PCS items, bypassing the state machine.
 */

#include <AssertMacros.h>
#include <Security/SecBasePriv.h>
#include <Security/SecItem.h>
#include <Security/SecItemFetchOutOfBandPriv.h>

#include <errno.h>
#include <limits.h>
#include <sqlite3.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <Security/SecBase.h>
#include <CoreFoundation/CFData.h>
#include <CoreFoundation/CFDate.h>
#include <CoreFoundation/CFDictionary.h>
#include <CoreFoundation/CFNumber.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFURL.h>
#include <CommonCrypto/CommonDigest.h>
#include <libkern/OSByteOrder.h>
#include <corecrypto/ccder.h>
#include <utilities/array_size.h>
#include <utilities/debugging.h>
#include <utilities/SecCFError.h>
#include <utilities/SecCFWrappers.h>
#include <utilities/SecIOFormat.h>
#include <utilities/SecXPCError.h>
#include <utilities/SecFileLocations.h>
#include <utilities/der_plist.h>
#include <utilities/der_plist_internal.h>
#include <utilities/simulatecrash_assert.h>
#include <libaks_acl_cf_keys.h>
#include <os/activity.h>
#include <pthread.h>
#include <os/lock.h>
#include <os/feature_private.h>

#include <Security/SecInternal.h>
#include "keychain/SecureObjectSync/SOSInternal.h"
#include <TargetConditionals.h>
#include <ipc/securityd_client.h>
#include <Security/SecuritydXPC.h>
#include <AssertMacros.h>
#include <asl.h>
#include <sys/types.h>
#include <pwd.h>
#include <grp.h>
#include <unistd.h>
#include <libDER/asn1Types.h>

#include <utilities/SecDb.h>
#include <IOKit/IOReturn.h>

#import <LocalAuthentication/LocalAuthentication_Private.h>
#import <CryptoTokenKit/CryptoTokenKit_Private.h>

#include "SecItemRateLimit.h"
#include "SecSoftLink.h"

#include <Security/OTConstants.h>


@implementation CKKSCurrentItemQuery
- (instancetype)initWithIdentifier:(NSString*)identifier
                       accessGroup:(NSString*)accessGroup
                            zoneID:(NSString*)zoneID
{
    if (self = [super init]) {
        _identifier = identifier;
        _accessGroup = accessGroup;
        _zoneID = zoneID;
    }
    return self;
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"<CKKSCurrentItemQuery(%@-%@): %@>",
            self.zoneID,
            self.accessGroup,
            self.identifier];
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder {
    [coder encodeObject:self.identifier forKey:@"identifier"];
    [coder encodeObject:self.accessGroup forKey:@"accessGroup"];
    [coder encodeObject:self.zoneID forKey:@"zoneID"];
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder {

    if ((self = [super init])) {
        _identifier = [coder decodeObjectOfClass:[NSString class] forKey:@"identifier"];
        _accessGroup = [coder decodeObjectOfClass:[NSString class] forKey:@"accessGroup"];
        _zoneID = [coder decodeObjectOfClass:[NSString class] forKey:@"zoneID"];
    }
    return self;
}

@end

@implementation CKKSCurrentItemQueryResult
- (instancetype)initWithIdentifier:(NSString*)identifier
                       accessGroup:(NSString*)accessGroup
                            zoneID:(NSString*)zoneID
                   decryptedRecord:(NSDictionary*)decryptedRecord
{
    if (self = [super init]) {
        _identifier = identifier;
        _accessGroup = accessGroup;
        _zoneID = zoneID;
        _decryptedRecord = decryptedRecord;
    }
    return self;
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"<CKKSCurrentItemQueryResult(%@-%@): %@>",
            self.zoneID,
            self.accessGroup,
            self.identifier];
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder {
    [coder encodeObject:self.identifier forKey:@"identifier"];
    [coder encodeObject:self.accessGroup forKey:@"accessGroup"];
    [coder encodeObject:self.zoneID forKey:@"zoneID"];
    [coder encodeObject:self.decryptedRecord forKey:@"decryptedRecord"];
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder {

    if ((self = [super init])) {
        _identifier = [coder decodeObjectOfClass:[NSString class] forKey:@"identifier"];
        _accessGroup = [coder decodeObjectOfClass:[NSString class] forKey:@"accessGroup"];
        _zoneID = [coder decodeObjectOfClass:[NSString class] forKey:@"zoneID"];

        NSSet *allowedClasses = [NSSet setWithArray:@[[NSString class], [NSNumber class], [NSData class], [NSDate class], [NSDictionary class]]];
        _decryptedRecord = [coder decodeObjectOfClasses:allowedClasses forKey:@"decryptedRecord"];

    }
    return self;
}

@end

@implementation CKKSPCSIdentityQuery
- (instancetype)initWithServiceNumber:(NSNumber*)serviceNumber
                          accessGroup:(NSString*)accessGroup
                            publicKey:(NSString*)publicKey
                               zoneID:(NSString*)zoneID
{
    if (self = [super init]) {
        _serviceNumber = serviceNumber;
        _accessGroup = accessGroup;
        _publicKey = publicKey;
        _zoneID = zoneID;
    }
    return self;
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"<CKKSPCSIdentityQuery(%@): %@>",
            self.zoneID,
            self.serviceNumber];
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder {
    [coder encodeObject:self.serviceNumber forKey:@"serviceNumber"];
    [coder encodeObject:self.accessGroup forKey:@"accessGroup"];
    [coder encodeObject:self.zoneID forKey:@"zoneID"];
    [coder encodeObject:self.publicKey forKey:@"publicKey"];
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder {

    if ((self = [super init])) {
        _serviceNumber = [coder decodeObjectOfClass:[NSNumber class] forKey:@"serviceNumber"];
        _accessGroup = [coder decodeObjectOfClass:[NSString class] forKey:@"accessGroup"];
        _zoneID = [coder decodeObjectOfClass:[NSString class] forKey:@"zoneID"];
        _publicKey = [coder decodeObjectOfClass:[NSString class] forKey:@"publicKey"];
    }
    return self;
}

@end

@implementation CKKSPCSIdentityQueryResult
- (instancetype)initWithServiceNumber:(NSNumber*)serviceNumber
                            publicKey:(NSString*)publicKey
                               zoneID:(NSString*)zoneID
                      decryptedRecord:(NSDictionary*)decryptedRecord
{
    if (self = [super init]) {
        _serviceNumber = serviceNumber;
        _publicKey = publicKey;
        _zoneID = zoneID;
        _decryptedRecord = decryptedRecord;
    }
    return self;
}

+ (BOOL)supportsSecureCoding {
    return YES;
}

- (NSString*)description {
    return [NSString stringWithFormat:@"<CKKSPCSIdentityQueryResult(%@): %@>",
            self.zoneID,
            self.serviceNumber];
}

- (void)encodeWithCoder:(nonnull NSCoder *)coder {
    [coder encodeObject:self.serviceNumber forKey:@"serviceNumber"];
    [coder encodeObject:self.zoneID forKey:@"zoneID"];
    [coder encodeObject:self.publicKey forKey:@"publicKey"];
    [coder encodeObject:self.decryptedRecord forKey:@"decryptedRecord"];
    
}

- (nullable instancetype)initWithCoder:(nonnull NSCoder *)coder {

    if ((self = [super init])) {
        _serviceNumber = [coder decodeObjectOfClass:[NSNumber class] forKey:@"serviceNumber"];
        _zoneID = [coder decodeObjectOfClass:[NSString class] forKey:@"zoneID"];
        _publicKey = [coder decodeObjectOfClass:[NSString class] forKey:@"publicKey"];

        NSSet *allowedClasses = [NSSet setWithArray:@[[NSString class], [NSNumber class], [NSData class], [NSDate class], [NSDictionary class]]];
        _decryptedRecord = [coder decodeObjectOfClasses:allowedClasses forKey:@"decryptedRecord"];

    }
    return self;
}

@end


void SecItemFetchCurrentItemOutOfBand(NSArray<CKKSCurrentItemQuery*>* currentItemQueries, bool forceFetch, void (^complete)(NSArray<CKKSCurrentItemQueryResult*>* currentItems, NSError* error)) {
    os_activity_t activity = os_activity_create("fetchCurrentItemOutOfBand", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_DEFAULT);
    os_activity_scope(activity);

    @autoreleasepool {
        id<SecuritydXPCProtocol> rpc = SecuritydXPCProxyObject(false, ^(NSError *error) {
            complete(nil, error);
        });

        [rpc secItemFetchCurrentItemOutOfBand:currentItemQueries
                                   forceFetch:forceFetch
                                     complete:^(NSArray* currentItems, NSError* operror) {
            complete(currentItems, operror);
        }];
    }
}

void SecItemFetchPCSIdentityOutOfBand(NSArray<CKKSPCSIdentityQuery*>* pcsIdentityQueries, bool forceFetch, void (^complete)(NSArray<CKKSPCSIdentityQueryResult*>* pcsIdentities, NSError* error)) {
    os_activity_t activity = os_activity_create("fetchPCSIdentityOutOfBand", OS_ACTIVITY_CURRENT, OS_ACTIVITY_FLAG_DEFAULT);
    os_activity_scope(activity);

    @autoreleasepool {
        id<SecuritydXPCProtocol> rpc = SecuritydXPCProxyObject(false, ^(NSError *error) {
            complete(nil, error);
        });
        
        [rpc secItemFetchPCSIdentityByKeyOutOfBand:pcsIdentityQueries
                                        forceFetch:forceFetch
                                          complete:^(NSArray* pcsIdentities, NSError* operror) {
            complete(pcsIdentities, operror);
        }];
    }
}
