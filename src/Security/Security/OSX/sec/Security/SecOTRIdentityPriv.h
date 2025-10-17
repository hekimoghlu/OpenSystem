/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#ifndef _SECOTRIDENTITYPRIV_H_

#include <CoreFoundation/CFRuntime.h>
#include <CoreFoundation/CFData.h>

#include <Security/SecKey.h>

#include <Security/oidsalg.h>

#include <CommonCrypto/CommonDigest.h> // DIGEST_LENGTH
#include <Security/SecOTR.h>

__BEGIN_DECLS
    
// OAEP Padding, uses lots of space. Might need this to be data
// Driven when we support more key types.
#define kPaddingOverhead (2 + 2 * CC_SHA1_DIGEST_LENGTH + 1)
    
//
// Identity opaque structs
//

#define kMPIDHashSize   CC_SHA1_DIGEST_LENGTH

struct _SecOTRFullIdentity {
    CFRuntimeBase _base;
    
    SecKeyRef   publicSigningKey;
    SecKeyRef   privateSigningKey;
    CFDataRef   privateKeyPersistentRef;

    bool        isMessageProtectionKey;
    uint8_t     publicIDHash[kMPIDHashSize];
};


struct _SecOTRPublicIdentity {
    CFRuntimeBase _base;
    
    SecKeyRef   publicSigningKey;

    bool        wantsHashes;

    uint8_t     hash[kMPIDHashSize];
};

enum SecOTRError {
    secOTRErrorLocal,
    secOTRErrorOSError,
};

extern const SecAsn1AlgId *kOTRSignatureAlgIDPtr;
void EnsureOTRAlgIDInited(void);
    
// Private functions for Public and Full IDs

bool SecOTRFIAppendSignature(SecOTRFullIdentityRef fullID,
                                CFDataRef dataToHash,
                                CFMutableDataRef appendTo,
                                CFErrorRef *error);

void SecOTRFIAppendPublicHash(SecOTRFullIdentityRef fullID, CFMutableDataRef appendTo);
bool SecOTRFIComparePublicHash(SecOTRFullIdentityRef fullID, const uint8_t hash[kMPIDHashSize]);

size_t SecOTRFISignatureSize(SecOTRFullIdentityRef privateID);

bool SecOTRFICompareToPublicKey(SecOTRFullIdentityRef fullID, SecKeyRef publicKey);

bool SecOTRPIVerifySignature(SecOTRPublicIdentityRef publicID,
                                const uint8_t *dataToHash, size_t amountToHash,
                                const uint8_t *signatureStart, size_t signatureSize, CFErrorRef *error);

bool SecOTRPIEqualToBytes(SecOTRPublicIdentityRef id, const uint8_t*bytes, CFIndex size);
bool SecOTRPIEqual(SecOTRPublicIdentityRef left, SecOTRPublicIdentityRef right);

size_t SecOTRPISignatureSize(SecOTRPublicIdentityRef publicID);
    
void SecOTRPICopyHash(SecOTRPublicIdentityRef publicID, uint8_t hash[kMPIDHashSize]);
void SecOTRPIAppendHash(SecOTRPublicIdentityRef publicID, CFMutableDataRef appendTo);

bool SecOTRPICompareHash(SecOTRPublicIdentityRef publicID, const uint8_t hash[kMPIDHashSize]);

bool SecOTRPICompareToPublicKey(SecOTRPublicIdentityRef publicID, SecKeyRef publicKey);


// Utility streaming functions
OSStatus insertSize(CFIndex size, uint8_t* here);
OSStatus appendSize(CFIndex size, CFMutableDataRef into);
OSStatus readSize(const uint8_t** data, size_t* limit, uint16_t* size);

OSStatus appendPublicOctets(SecKeyRef fromKey, CFMutableDataRef appendTo);
OSStatus appendPublicOctetsAndSize(SecKeyRef fromKey, CFMutableDataRef appendTo);
OSStatus appendSizeAndData(CFDataRef data, CFMutableDataRef appendTo);

SecKeyRef CreateECPublicKeyFrom(CFAllocatorRef allocator, const uint8_t** data, size_t* limit);
    
bool SecOTRCreateError(enum SecOTRError family, CFIndex errorCode, CFStringRef descriptionString, CFErrorRef previousError, CFErrorRef *newError);

__END_DECLS

#endif
