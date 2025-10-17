/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#ifndef _BITMAP_H
#define _BITMAP_H

#include <sys/types.h>

/* Simple bit vector routines */

struct bitmap;

/* Allocate a new bitmap. Returns NULL on allocation failure. */
struct bitmap *bitmap_new(void);

/* Free a bitmap */
void bitmap_free(struct bitmap *b);

/* Zero an existing bitmap */
void bitmap_zero(struct bitmap *b);

/* Test whether a bit is set in a bitmap. */
int bitmap_test_bit(struct bitmap *b, u_int n);

/* Set a bit in a bitmap. Returns 0 on success or -1 on error */
int bitmap_set_bit(struct bitmap *b, u_int n);

/* Clear a bit in a bitmap */
void bitmap_clear_bit(struct bitmap *b, u_int n);

/* Return the number of bits in a bitmap (i.e. the position of the MSB) */
size_t bitmap_nbits(struct bitmap *b);

/* Return the number of bytes needed to represent a bitmap */
size_t bitmap_nbytes(struct bitmap *b);

/* Convert a bitmap to a big endian byte string */
int bitmap_to_string(struct bitmap *b, void *p, size_t l);

/* Convert a big endian byte string to a bitmap */
int bitmap_from_string(struct bitmap *b, const void *p, size_t l);

#endif /* _BITMAP_H */
