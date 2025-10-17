/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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
#ifndef OPENSSL_HEADER_CRYPTO_SPX_UTIL_H
#define OPENSSL_HEADER_CRYPTO_SPX_UTIL_H

#include <openssl/base.h>

#if defined(__cplusplus)
extern "C" {
#endif


// Encodes the integer value of input to out_len bytes in big-endian order.
// Note that input < 2^(8*out_len), as otherwise this function will truncate
// the least significant bytes of the integer representation.
void spx_uint64_to_len_bytes(uint8_t *output, size_t out_len, uint64_t input);

uint64_t spx_to_uint64(const uint8_t *input, size_t input_len);

// Compute the base 2^log2_b representation of X.
//
// As some of the parameter sets in https://eprint.iacr.org/2022/1725.pdf use
// a FORS height > 16 we use a uint32_t to store the output.
void spx_base_b(uint32_t *output, size_t out_len, const uint8_t *input,
                unsigned int log2_b);


#if defined(__cplusplus)
}  // extern C
#endif

#endif  // OPENSSL_HEADER_CRYPTO_SPX_UTIL_H
