/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#import "OTPrivateKey+SF.h"
#import <SecurityFoundation/SFKey_Private.h>
#import "keychain/ot/OTDefines.h"
#import "keychain/ot/OTConstants.h"
#import "utilities/SecCFWrappers.h"

@implementation OTPrivateKey (SecurityFoundation)

+ (instancetype)fromECKeyPair:(SFECKeyPair *)keyPair
{
    OTPrivateKey *pk = [OTPrivateKey new];
    pk.keyType = OTPrivateKey_KeyType_EC_NIST_CURVES;
    pk.keyData = keyPair.keyData;
    return pk;
}

+ (SecKeyRef) createSecKey:(NSData*)keyData CF_RETURNS_RETAINED
{
    NSDictionary *keyAttributes = @{
                                    (__bridge id)kSecAttrKeyClass : (__bridge id)kSecAttrKeyClassPrivate,
                                    (__bridge id)kSecAttrKeyType : (__bridge id)kSecAttrKeyTypeEC,
                                    };

    SecKeyRef key = SecKeyCreateWithData((__bridge CFDataRef)keyData, (__bridge CFDictionaryRef)keyAttributes, NULL);
    return key;
}

- (SFECKeyPair * _Nullable)asECKeyPair:(NSError**)error
{
    if (self.keyType != OTPrivateKey_KeyType_EC_NIST_CURVES) {
        if(error){
            *error = [NSError errorWithDomain:OctagonErrorDomain code:OctagonErrorNotSupported userInfo:nil];
        }
        return nil;
    }
    SecKeyRef secKey = [OTPrivateKey createSecKey:self.keyData];
    SFECKeyPair *result = [[SFECKeyPair alloc] initWithSecKey:secKey];
    CFReleaseNull(secKey);
    return result;
}

@end

