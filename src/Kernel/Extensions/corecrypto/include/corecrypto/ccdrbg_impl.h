/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

/* opaque drbg structure */
struct ccdrbg_state;

struct ccdrbg_info {
    /*! Size of the DRBG state in bytes **/
    size_t size;

    /*! Instantiate the PRNG
     @param prng       The PRNG state
     @param entropylen Length of entropy
     @param entropy    Entropy bytes
     @param inlen      Length of additional input
     @param in         Additional input bytes
     @return 0 if successful
     */
    int (*init)(const struct ccdrbg_info *info, struct ccdrbg_state *drbg,
                size_t entropyLength, const void* entropy,
                size_t nonceLength, const void* nonce,
                size_t psLength, const void* ps);

    /*! Add entropy to the PRNG
     @param prng       The PRNG state
     @param entropylen Length of entropy
     @param entropy    Entropy bytes
     @param inlen      Length of additional input
     @param in         Additional input bytes
     @return 0 if successful
     */
    int (*reseed)(struct ccdrbg_state *prng,
                  size_t entropylen, const void *entropy,
                  size_t inlen, const void *in);

    /*! Read from the PRNG in a FIPS Testing compliant manor
     @param prng    The PRNG state to read from
     @param out     [out] Where to store the data
     @param outlen  Length of data desired (octets)
     @param inlen   Length of additional input
     @param in      Additional input bytes
     @return 0 if successfull
     */
    int (*generate)(struct ccdrbg_state *prng,
                    size_t outlen, void *out,
                    size_t inlen, const void *in);

    /*! Terminate a PRNG state
     @param prng   The PRNG state to terminate
     */
    void (*done)(struct ccdrbg_state *prng);

    /** private parameters */
    const void *custom;
};



#endif // _CORECRYPTO_CCDRBG_IMPL_H_
