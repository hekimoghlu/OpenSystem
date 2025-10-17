/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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

#import "keychain/TrustedPeersHelper/TPHObjcTranslation.h"
#include "utilities/SecCFRelease.h"

#import <Security/SecKey.h>
#import <Security/SecKeyPriv.h>

#import <SecurityFoundation/SFKey.h>
#import <SecurityFoundation/SFKey_Private.h>
#import <corecrypto/ccsha2.h>
#import <corecrypto/ccrng.h>

@implementation TPHObjectiveC : NSObject

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

+ (SFECKeyPair* _Nullable)fetchKeyPairWithPrivateKeyPersistentRef:(NSData *)persistentRef error:(NSError**)error
{
    NSDictionary* query = @{
        (id)kSecReturnRef : @YES,
        (id)kSecClass : (id)kSecClassKey,
        (id)kSecValuePersistentRef : persistentRef,
    };

    CFTypeRef foundRef = NULL;
    OSStatus status = SecItemCopyMatching((__bridge CFDictionaryRef)query, &foundRef);

    if (status == errSecSuccess && CFGetTypeID(foundRef) == SecKeyGetTypeID()) {
        SFECKeyPair* keyPair = [[SFECKeyPair alloc] initWithSecKey:(SecKeyRef)foundRef];
        CFReleaseNull(foundRef);
        return keyPair;
    } else {
        if(error) {
            *error = [NSError errorWithDomain:NSOSStatusErrorDomain
                                         code:status
                                     userInfo:nil];
        }
        return nil;
    }
}

#pragma clang diagnostic pop

+ (ccec_full_ctx_t)ccec384Context
{
    ccec_const_cp_t cp = ccec_cp_384();
    size_t size = ccec_full_ctx_size(ccec_ccn_size(cp));
    ccec_full_ctx_t heapContext = (ccec_full_ctx_t)malloc(size);
    ccec_ctx_init(cp, heapContext);
    return heapContext;
}

+ (void)contextFree:(void*) context
{
    free(context);
}

+ (size_t) ccsha384_diSize{
    return ccsha384_di()->output_size;
}

+ (SFAESKeyBitSize)aes256BitSize{
    return SFAESKeyBitSize256;
}

+ (NSString*)digestUsingSha384:(NSData*) data {
    const struct ccdigest_info *di = ccsha384_di();
    NSMutableData* result = [[NSMutableData alloc] initWithLength:ccsha384_di()->output_size];

    ccdigest(di, [data length], [data bytes], [result mutableBytes]);

    return [result base64EncodedStringWithOptions:0];
}

@end
