/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#include <openssl/base.h>

#include <assert.h>

#include "./spx_util.h"

void spx_uint64_to_len_bytes(uint8_t *output, size_t out_len, uint64_t input) {
  for (size_t i = out_len; i > 0; --i) {
    output[i - 1] = input & 0xff;
    input = input >> 8;
  }
}

uint64_t spx_to_uint64(const uint8_t *input, size_t input_len) {
  uint64_t tmp = 0;
  for (size_t i = 0; i < input_len; ++i) {
    tmp = 256 * tmp + input[i];
  }
  return tmp;
}

void spx_base_b(uint32_t *output, size_t out_len, const uint8_t *input,
                unsigned int log2_b) {
  int in = 0;
  uint32_t out = 0;
  uint32_t bits = 0;
  uint32_t total = 0;
  uint32_t base = UINT32_C(1) << log2_b;

  for (out = 0; out < out_len; ++out) {
    while (bits < log2_b) {
      total = (total << 8) + input[in];
      in++;
      bits = bits + 8;
    }
    bits -= log2_b;
    output[out] = (total >> bits) % base;
  }
}
