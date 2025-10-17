/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 16, 2021.
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
/* direct-mapped partial matching compressor with simple 22/10 split
 *
 *  Compresses buffers using a dictionary based match and partial match
 *  (high bits only or full match) scheme.
 *
 *  Paul Wilson -- wilson@cs.utexas.edu
 *  Scott F. Kaplan -- sfkaplan@cs.utexas.edu
 *  September 1997
 */

/* compressed output format, in memory order
 *  1. a four-word HEADER containing four one-word values:
 *     i.   a one-word code saying what algorithm compressed the data
 *     ii.  an integer WORD offset into the page saying
 *          where the queue position area starts
 *     iii. an integer WORD offset into the page saying where
 *          the low-bits area starts
 *     iv.  an integer WORD offset into the page saying where the
 *          low-bits area ends
 *
 *  2. a 64-word TAGS AREA holding one two-bit tag for each word in
 *     the original (1024-word) page, packed 16 per word
 *
 *  3. a variable-sized FULL WORDS AREA (always word aligned and an
 *     integral number of words) holding full-word patterns that
 *     were not in the dictionary when encoded (i.e., dictionary misses)
 *
 *  4. a variable-sized QUEUE POSITIONS AREA (always word aligned and
 *     an integral number of words) holding four-bit queue positions,
 *     packed eight per word.
 *
 *  5. a variable-sized LOW BITS AREA (always word aligned and an
 *     integral number of words) holding ten-bit low-bit patterns
 *     (from partial matches), packed three per word.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include <mach/vm_param.h>


#define WKdm_SCRATCH_BUF_SIZE_INTERNAL  PAGE_SIZE

typedef unsigned int WK_word;

#if defined(__arm64__)

void
WKdm_decompress_4k(const WK_word* src_buf,
    WK_word* dest_buf,
    WK_word* scratch,
    unsigned int bytes);
int
WKdm_compress_4k(const WK_word* src_buf,
    WK_word* dest_buf,
    WK_word* scratch,
    unsigned int limit);

void
WKdm_decompress_16k(WK_word* src_buf,
    WK_word* dest_buf,
    WK_word* scratch,
    unsigned int bytes);
int
WKdm_compress_16k(WK_word* src_buf,
    WK_word* dest_buf,
    WK_word* scratch,
    unsigned int limit);
#else

void
WKdm_decompress_new(WK_word* src_buf,
    WK_word* dest_buf,
    WK_word* scratch,
    unsigned int bytes);
int
WKdm_compress_new(const WK_word* src_buf,
    WK_word* dest_buf,
    WK_word* scratch,
    unsigned int limit);
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
