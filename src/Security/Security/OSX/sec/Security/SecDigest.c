/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 1, 2023.
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
#ifdef STANDALONE
/* Allows us to build genanchors against the BaseSDK. */
#undef __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__
#undef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
#endif

#include "SecFramework.h"
#include <dispatch/dispatch.h>
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonDigestSPI.h>
#include <Security/SecAsn1Coder.h>
#include <Security/oidsalg.h>
#include <utilities/SecCFWrappers.h>
#include <Security/SecBase.h>
#include <inttypes.h>

/* Return the SHA1 digest of a chunk of data as newly allocated CFDataRef. */
CFDataRef SecSHA1DigestCreate(CFAllocatorRef allocator,
                              const UInt8 *data, CFIndex length) {
    if (length < 0 || length > INT32_MAX || data == NULL) {
        return NULL; /* guard against passing bad values to digest function */
    }
    CFMutableDataRef digest = CFDataCreateMutable(allocator,
                                                  CC_SHA1_DIGEST_LENGTH);
    CFDataSetLength(digest, CC_SHA1_DIGEST_LENGTH);
    CCDigest(kCCDigestSHA1, data, length, CFDataGetMutableBytePtr(digest));
    return digest;
}

CFDataRef SecSHA256DigestCreate(CFAllocatorRef allocator,
                                const UInt8 *data, CFIndex length) {
    if (length < 0 || length > INT32_MAX || data == NULL) {
        return NULL; /* guard against passing bad values to digest function */
    }
    CFMutableDataRef digest = CFDataCreateMutable(allocator,
                                                  CC_SHA256_DIGEST_LENGTH);
    CFDataSetLength(digest, CC_SHA256_DIGEST_LENGTH);
    CCDigest(kCCDigestSHA256, data, length, CFDataGetMutableBytePtr(digest));
    return digest;
}

CFDataRef SecSHA256DigestCreateFromData(CFAllocatorRef allocator, CFDataRef data) {
    CFMutableDataRef digest = CFDataCreateMutable(allocator,
                                                  CC_SHA256_DIGEST_LENGTH);
    CFDataSetLength(digest, CC_SHA256_DIGEST_LENGTH);
    CCDigest(kCCDigestSHA256, CFDataGetBytePtr(data), CFDataGetLength(data), CFDataGetMutableBytePtr(digest));
    return digest;
}

CFDataRef SecDigestCreate(CFAllocatorRef allocator,
                          const SecAsn1Oid *algorithm, const SecAsn1Item *params,
                          const UInt8 *data, CFIndex length) {
    unsigned char *(*digestFcn)(const void *data, CC_LONG len, unsigned char *md);
    CFIndex digestLen;

    if (length < 0 || length > INT32_MAX || data == NULL)
        return NULL; /* guard against passing bad values to digest function */

    if (SecAsn1OidCompare(algorithm, &CSSMOID_SHA1)) {
        digestFcn = CC_SHA1;
        digestLen = CC_SHA1_DIGEST_LENGTH;
    } else if (SecAsn1OidCompare(algorithm, &CSSMOID_SHA224)) {
        digestFcn = CC_SHA224;
        digestLen = CC_SHA224_DIGEST_LENGTH;
    } else if (SecAsn1OidCompare(algorithm, &CSSMOID_SHA256)) {
        digestFcn = CC_SHA256;
        digestLen = CC_SHA256_DIGEST_LENGTH;
    } else if (SecAsn1OidCompare(algorithm, &CSSMOID_SHA384)) {
        digestFcn = CC_SHA384;
        digestLen = CC_SHA384_DIGEST_LENGTH;
    } else if (SecAsn1OidCompare(algorithm, &CSSMOID_SHA512)) {
        digestFcn = CC_SHA512;
        digestLen = CC_SHA512_DIGEST_LENGTH;
    } else {
        return NULL;
    }

    CFMutableDataRef digest = CFDataCreateMutable(allocator, digestLen);
    CFDataSetLength(digest, digestLen);

    digestFcn(data, (CC_LONG)length, CFDataGetMutableBytePtr(digest));
    return digest;
}
