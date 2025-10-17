/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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
#ifndef __BITARRAY_H
#define __BITARRAY_H

typedef uint64_t *bitarray_t; // array of bits, assumed to be mostly 0
typedef uint32_t index_t; // we limit the number of bits to be a 32-bit quantity

/* A bitarray uses a summarization to be able to quickly say what's the first bit that is set to 1;
 All together each of the entry points will do a very small number of memory access (exact number depends on log_size) */

MALLOC_NOEXPORT extern size_t bitarray_size(unsigned log_size);
    // For a bitarray with 1<<log_size bits, returns the number of bytes needed to contain the whole bitarray

MALLOC_NOEXPORT extern bitarray_t bitarray_create(unsigned log_size);
    // creates a bitarray with 1<<log_size bits, all initialized to 0.  
    // Use free() to free bitarray

MALLOC_NOEXPORT extern bool bitarray_get(bitarray_t bits, unsigned log_size, index_t index);

MALLOC_NOEXPORT extern bool bitarray_set(bitarray_t bits, unsigned log_size, index_t index);
    // Set a bit in bitarray

MALLOC_NOEXPORT extern bool bitarray_zap(bitarray_t bits, unsigned log_size, index_t index);
    // Clears a bit in bitarray

MALLOC_NOEXPORT extern index_t bitarray_first_set(const bitarray_t bits, unsigned log_size);
    // Returns the index first bit that's 1, plus 1, or 0 if all the bits are zero

MALLOC_NOEXPORT extern bool bitarray_zap_first_set(bitarray_t bits, unsigned log_size, index_t *index);
    // finds the first bit set, and if found, zaps it and sets index

MALLOC_NOEXPORT extern unsigned bitarray_zap_first_set_multiple(bitarray_t bits, unsigned log_size, unsigned max, index_t *indices);
    // finds all the bits set, up to max, and zaps each and sets the index for each
    // returns number zapped

static MALLOC_INLINE MALLOC_ALWAYS_INLINE void
BITARRAY_SET(uint32_t *bits, msize_t index)
{
	// index >> 5 identifies the uint32_t to manipulate in the conceptually contiguous bits array
	// (index >> 5) << 1 identifies the uint32_t allowing for the actual interleaving
	bits[(index >> 5) << 1] |= (1 << (index & 31));
}

static MALLOC_INLINE MALLOC_ALWAYS_INLINE void
BITARRAY_CLR(uint32_t *bits, msize_t index)
{
	bits[(index >> 5) << 1] &= ~(1 << (index & 31));
}

static MALLOC_INLINE MALLOC_ALWAYS_INLINE boolean_t
BITARRAY_BIT(uint32_t *bits, msize_t index)
{
	return ((bits[(index >> 5) << 1]) >> (index & 31)) & 1;
}

/* Macros used to manipulate the uint32_t quantity mag_bitmap. */

/* BITMAPV variants are used by tiny. */
#if defined(__LP64__)
// assert(NUM_SLOTS == 64) in which case (slot >> 5) is either 0 or 1
#define BITMAPV_SET(bitmap, slot) (bitmap[(slot) >> 5] |= 1 << ((slot)&31))
#define BITMAPV_CLR(bitmap, slot) (bitmap[(slot) >> 5] &= ~(1 << ((slot)&31)))
#define BITMAPV_BIT(bitmap, slot) ((bitmap[(slot) >> 5] >> ((slot)&31)) & 1)
#define BITMAPV_CTZ(bitmap) (__builtin_ctzl(bitmap))
#else
// assert(NUM_SLOTS == 32) in which case (slot >> 5) is always 0, so code it that way
#define BITMAPV_SET(bitmap, slot) (bitmap[0] |= 1 << (slot))
#define BITMAPV_CLR(bitmap, slot) (bitmap[0] &= ~(1 << (slot)))
#define BITMAPV_BIT(bitmap, slot) ((bitmap[0] >> (slot)) & 1)
#define BITMAPV_CTZ(bitmap) (__builtin_ctz(bitmap))
#endif

/* BITMAPN is used by small. (slot >> 5) takes on values from 0 to 7. */
#define BITMAPN_SET(bitmap, slot) (bitmap[(slot) >> 5] |= 1 << ((slot)&31))
#define BITMAPN_CLR(bitmap, slot) (bitmap[(slot) >> 5] &= ~(1 << ((slot)&31)))
#define BITMAPN_BIT(bitmap, slot) ((bitmap[(slot) >> 5] >> ((slot)&31)) & 1)

/* returns bit # of least-significant one bit, starting at 0 (undefined if !bitmap) */
#define BITMAP32_CTZ(bitmap) (__builtin_ctz(bitmap[0]))

#endif // __BITARRAY_H
