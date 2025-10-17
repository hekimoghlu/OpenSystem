/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
    @header     CommonHMAC.h
    @abstract   Keyed Message Authentication Code (HMAC) functions.
 */

#ifndef _CC_COMMON_HMAC_H_
#define _CC_COMMON_HMAC_H_

#include <CommonCrypto/CommonDigest.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
    @enum       CCHmacAlgorithm
    @abstract   Algorithms implemented in this module.

    @constant   kCCHmacAlgSHA1        HMAC with SHA1 digest
    @constant   kCCHmacAlgMD5          HMAC with MD5 digest
    @constant   kCCHmacAlgSHA256    HMAC with SHA256 digest
    @constant   kCCHmacAlgSHA384    HMAC with SHA384 digest
    @constant   kCCHmacAlgSHA512    HMAC with SHA512 digest
    @constant   kCCHmacAlgSHA224    HMAC with SHA224 digest
 */
enum {
    kCCHmacAlgSHA1,
    kCCHmacAlgMD5,
    kCCHmacAlgSHA256,
    kCCHmacAlgSHA384,
    kCCHmacAlgSHA512,
    kCCHmacAlgSHA224
};
typedef uint32_t CCHmacAlgorithm;

/*!
    @typedef    CCHmacContext
    @abstract   HMAC context.
 */
#define CC_HMAC_CONTEXT_SIZE    96
typedef struct {
    uint32_t            ctx[CC_HMAC_CONTEXT_SIZE];
} CCHmacContext;

/*!
    @function   CCHmacInit
    @abstract   Initialize an CCHmacContext with provided raw key bytes.

    @param      ctx         An HMAC context.
    @param      algorithm   HMAC algorithm to perform.
    @param      key         Raw key bytes.
    @param      keyLength   Length of raw key bytes; can be any
                            length including zero.
 */
void CCHmacInit(
    CCHmacContext *ctx,
    CCHmacAlgorithm algorithm,
    const void *key,
    size_t keyLength)
    API_AVAILABLE(macos(10.4), ios(2.0));


/*!
    @function   CCHmacUpdate
    @abstract   Process some data.

    @param      ctx         An HMAC context.
    @param      data        Data to process.
    @param      dataLength  Length of data to process, in bytes.

    @discussion This can be called multiple times.
 */
void CCHmacUpdate(
    CCHmacContext *ctx,
    const void *data,
    size_t dataLength)
    API_AVAILABLE(macos(10.4), ios(2.0));


/*!
    @function   CCHmacFinal
    @abstract   Obtain the final Message Authentication Code.

    @param      ctx         An HMAC context.
    @param      macOut      Destination of MAC; allocated by caller.

    @discussion The length of the MAC written to *macOut is the same as
                the digest length associated with the HMAC algorithm:

                kCCHmacAlgSHA1 : CC_SHA1_DIGEST_LENGTH
                kCCHmacAlgSHA256  : CC_SHA256_DIGEST_LENGTH
 
                The MAC must be verified by comparing the computed and expected values
                using timingsafe_bcmp. Other comparison functions (e.g. memcmp)
                must not be used as they may be vulnerable to practical timing attacks,
                leading to MAC forgery.
 */
void CCHmacFinal(
    CCHmacContext *ctx,
    void *macOut)
    API_AVAILABLE(macos(10.4), ios(2.0));

/*!
     @function   CCHmac
     @abstract   Stateless, one-shot HMAC function
     
     @param      algorithm   HMAC algorithm to perform.
     @param      key         Raw key bytes.
     @param      keyLength   Length of raw key bytes; can be any
     length including zero.
     @param      data        Data to process.
     @param      dataLength  Length of data to process, in bytes.
     @param      macOut      Destination of MAC; allocated by caller.
     
     @discussion The length of the MAC written to *macOut is the same as the digest length associated with the HMAC algorithm:
                  kCCHmacAlgSHA1 : CC_SHA1_DIGEST_LENGTH
                  kCCHmacAlgSHA256  : CC_SHA256_DIGEST_LENGTH
     
                 The MAC must be verified by comparing the computed and expected values
                 using timingsafe_bcmp. Other comparison functions (e.g. memcmp)
                 must not be used as they may be vulnerable to practical timing attacks,
                 leading to MAC forgery.
*/
    
void CCHmac(
    CCHmacAlgorithm algorithm,  /* kCCHmacAlgSHA256, kCCHmacAlgSHA1 */
    const void *key,
    size_t keyLength,           /* length of key in bytes */
    const void *data,
    size_t dataLength,          /* length of data in bytes */
    void *macOut)               /* MAC written here */
    API_AVAILABLE(macos(10.4), ios(2.0));

#ifdef __cplusplus
}
#endif

#endif  /* _CC_COMMON_HMAC_H_ */
