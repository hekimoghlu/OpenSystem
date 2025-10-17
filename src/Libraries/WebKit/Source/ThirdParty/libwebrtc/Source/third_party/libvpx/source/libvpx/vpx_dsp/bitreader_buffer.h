/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 1, 2024.
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
#ifndef VPX_VPX_DSP_BITREADER_BUFFER_H_
#define VPX_VPX_DSP_BITREADER_BUFFER_H_

#include <limits.h>

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*vpx_rb_error_handler)(void *data);

struct vpx_read_bit_buffer {
  const uint8_t *bit_buffer;
  const uint8_t *bit_buffer_end;
  size_t bit_offset;

  void *error_handler_data;
  vpx_rb_error_handler error_handler;
};

size_t vpx_rb_bytes_read(struct vpx_read_bit_buffer *rb);

int vpx_rb_read_bit(struct vpx_read_bit_buffer *rb);

int vpx_rb_read_literal(struct vpx_read_bit_buffer *rb, int bits);

int vpx_rb_read_signed_literal(struct vpx_read_bit_buffer *rb, int bits);

int vpx_rb_read_inv_signed_literal(struct vpx_read_bit_buffer *rb, int bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_BITREADER_BUFFER_H_
