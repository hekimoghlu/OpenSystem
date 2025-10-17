/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#ifndef _CORECRYPTO_CCDRBG_IMPL_H_
#define _CORECRYPTO_CCDRBG_IMPL_H_

#include <corecrypto/cc.h>

/* opaque drbg structure */
struct ccdrbg_state;

struct ccdrbg_info {
    /*! Size of the DRBG state in bytes **/
    size_t size;

    /*! Instantiate the DRBG
     @param drbg       The DRBG state
     @param entropylen Length of entropy
     @param entropy    Entropy bytes
     @param inlen      Length of additional input
     @param in         Additional input bytes
     @return 0 if successful
     */
    int (*CC_SPTR(ccdrbg_info, init))(const struct ccdrbg_info *info, struct ccdrbg_state *drbg,
                size_t entropyLength, const void* entropy,
                size_t nonceLength, const void* nonce,
                size_t psLength, const void* ps);

    /*! Add entropy to the DRBG
     @param drbg       The DRBG state
     @param entropylen Length of entropy
     @param entropy    Entropy bytes
     @param inlen      Length of additional input
     @param in         Additional input bytes
     @return 0 if successful
     */
    int (*CC_SPTR(ccdrbg_info, reseed))(struct ccdrbg_state *drbg,
                  size_t entropylen, const void *entropy,
                  size_t inlen, const void *in);

    /*! Read from the DRBG in a FIPS Testing compliant manor
     @param drbg    The DRBG state to read from
     @param out     [out] Where to store the data
     @param outlen  Length of data desired (octets)
     @param inlen   Length of additional input
     @param in      Additional input bytes
     @return 0 if successfull
     */
    int (*CC_SPTR(ccdrbg_info, generate))(struct ccdrbg_state *drbg,
                    size_t outlen, void *out,
                    size_t inlen, const void *in);

    /*! Terminate a DRBG state
     @param drbg   The DRBG state to terminate
     */
    void (*CC_SPTR(ccdrbg_info, done))(struct ccdrbg_state *drbg);

    /** private parameters */
    const void *custom;

    /*! Whether the DRBG requires a reseed to continue generation
     @param drbg    The DRBG state
     @return true if the DRBG requires reseed; false otherwise
     */
    bool (*CC_SPTR(ccdrbg_info, must_reseed))(const struct ccdrbg_state *drbg);
};



#endif // _CORECRYPTO_CCDRBG_IMPL_H_
