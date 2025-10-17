/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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
#ifndef	_CC_HmacSPI_H_
#define _CC_HmacSPI_H_

#include <CommonCrypto/CommonHMAC.h>
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonDigestSPI.h>

#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef CCHmacContext * CCHmacContextRef;

CCHmacContextRef
CCHmacCreate(CCDigestAlg alg, const void *key, size_t keyLength)
API_AVAILABLE(macos(10.7), ios(5.0));

/* Create a clone of an initialized CCHmacContextRef - you must do this before use.  */
CCHmacContextRef
CCHmacClone(CCHmacContextRef ctx)
API_AVAILABLE(macos(10.10), ios(7.0));

/* Update and Final are reused from existing api, type changed from struct CCHmacContext * to CCHmacContextRef though */

void
CCHmacDestroy(CCHmacContextRef ctx)
API_AVAILABLE(macos(10.7), ios(5.0));

size_t
CCHmacOutputSizeFromRef(CCHmacContextRef ctx)
API_AVAILABLE(macos(10.7), ios(5.0));


size_t
CCHmacOutputSize(CCDigestAlg alg)
API_AVAILABLE(macos(10.7), ios(5.0));

/*
 * Stateless, one-shot HMAC function using digest constants
 * Output is written to caller-supplied buffer, as in CCHmacFinal().

 *
 * The tag must be verified by comparing the computed and expected values
 * using timingsafe_bcmp. Other comparison functions (e.g. memcmp)
 * must not be used as they may be vulnerable to practical timing attacks,
 * leading to tag forgery.
 */
void CCHmacOneShot(
            CCDigestAlg alg,  /* kCCHmacAlgSHA1, kCCHmacAlgMD5 */
            const void *key,
            size_t keyLength,           /* length of key in bytes */
            const void *data,
            size_t dataLength,          /* length of data in bytes */
            void *macOut)               /* MAC written here */
API_AVAILABLE(macos(10.10), ios(7.0));



#ifdef __cplusplus
}
#endif

#endif /* _CC_HmacSPI_H_ */
