/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#ifndef _CORECRYPTO_CCHKDF_H_
#define _CORECRYPTO_CCHKDF_H_

#include <corecrypto/ccdigest.h>

/*!
    @function			cchkdf
    @abstract			Perform a RFC5869-compliant HKDF.
                        https://tools.ietf.org/html/rfc5869
    @discussion			Derives output key data from input key data, optional salt, and info.

    @param di			Hash function to use.
    @param ikm_nbytes	Input key material length in bytes
    @param ikm			Input key material used to derive the new key
    @param salt_nbytes	Salt length length
    @param salt			Salt data
    @param info_nbytes	Info string length
    @param info			Info string
    @param dk_nbytes	Derived Key Length in bytes.
    @param dk			Derived key buffer to receive results of KDF

    @return 0 on success, non-zero on failure. See cc_error.h for more details.
 */

int cchkdf(const struct ccdigest_info *di,
           size_t ikm_nbytes,
           const void *ikm,
           size_t salt_nbytes,
           const void *salt,
           size_t info_nbytes,
           const void *info,
           size_t dk_nbytes,
           void *dk);

/*!
    @function			cchkdf_extract
    @abstract			Perform a RFC5869-compliant HKDF-Extract
                        https://tools.ietf.org/html/rfc5869
    @discussion			Extract a pseudorandom key (PRK) from input keying material and a salt.

                        Note: In most cases, clients should use `cchkdf`. This function
                        is only exposed for specific use cases.

    @param di			Hash function to use.
    @param salt_nbytes	Salt length length
    @param salt			Salt data
    @param ikm_nbytes	Input key material length in bytes
    @param ikm			Input key material used to derive the new key
    @param prk			Pseudorandom Key (PRK) buffer to receive results of KDF, which must be equal
                        to the size of the hash function (di).

    @return 0 on success, non-zero on failure. See cc_error.h for more details.
 */

int cchkdf_extract(const struct ccdigest_info *di,
                   size_t salt_nbytes,
                   const void *salt,
                   size_t ikm_nbytes,
                   const void *ikm,
                   void *prk);

/*!
    @function			cchkdf_expand
    @abstract			Perform a RFC5869-compliant HKDF-Expand
                        https://tools.ietf.org/html/rfc5869
    @discussion			Expands a pseudo-random key to the desired length, using the according
   info.

                        Note: In most cases, clients should use `cchkdf`. This function
                        is only exposed for specific use cases.

    @param di			Hash function to use.
    @param prk_nbytes	Pseudo-random key length in bytes
    @param prk			Pseudo-random key used to derive the new key
    @param info_nbytes	Info string length
    @param info			Info string
    @param dk_nbytes	Derived Key Length in bytes.
    @param dk			Derived key buffer to receive results of KDF

    @return 0 on success, non-zero on failure. See cc_error.h for more details.
 */

int cchkdf_expand(const struct ccdigest_info *di,
                  size_t prk_nbytes,
                  const void *prk,
                  size_t info_nbytes,
                  const void *info,
                  size_t dk_nbytes,
                  void *dk);

#endif /* _CORECRYPTO_CCHKDF_H_ */
