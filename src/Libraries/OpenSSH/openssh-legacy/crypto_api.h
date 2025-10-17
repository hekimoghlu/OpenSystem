/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
 * Assembled from generated headers and source files by Markus Friedl.
 * Placed in the public domain.
 */

#ifndef crypto_api_h
#define crypto_api_h

#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif
#include <stdlib.h>

typedef int32_t crypto_int32;
typedef uint32_t crypto_uint32;

#define randombytes(buf, buf_len) arc4random_buf((buf), (buf_len))

#define crypto_hashblocks_sha512_STATEBYTES 64U
#define crypto_hashblocks_sha512_BLOCKBYTES 128U

int	crypto_hashblocks_sha512(unsigned char *, const unsigned char *,
     unsigned long long);

#define crypto_hash_sha512_BYTES 64U

int	crypto_hash_sha512(unsigned char *, const unsigned char *,
    unsigned long long);

int	crypto_verify_32(const unsigned char *, const unsigned char *);

#define crypto_sign_ed25519_SECRETKEYBYTES 64U
#define crypto_sign_ed25519_PUBLICKEYBYTES 32U
#define crypto_sign_ed25519_BYTES 64U

int	crypto_sign_ed25519(unsigned char *, unsigned long long *,
    const unsigned char *, unsigned long long, const unsigned char *);
int	crypto_sign_ed25519_open(unsigned char *, unsigned long long *,
    const unsigned char *, unsigned long long, const unsigned char *);
int	crypto_sign_ed25519_keypair(unsigned char *, unsigned char *);

#endif /* crypto_api_h */
