/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "initarray.h"

static void
init_genrand(rk_state *self, unsigned long s);

/* initializes mt[RK_STATE_LEN] with a seed */
static void
init_genrand(rk_state *self, unsigned long s)
{
    int mti;
    unsigned long *mt = self->key;

    mt[0] = s & 0xffffffffUL;
    for (mti = 1; mti < RK_STATE_LEN; mti++) {
        /*
         * See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
         * In the previous versions, MSBs of the seed affect
         * only MSBs of the array mt[].
         * 2002/01/09 modified by Makoto Matsumoto
         */
        mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        /* for > 32 bit machines */
        mt[mti] &= 0xffffffffUL;
    }
    self->pos = mti;
    return;
}


/*
 * initialize by an array with array-length
 * init_key is the array for initializing keys
 * key_length is its length
 */
extern void
init_by_array(rk_state *self, unsigned long init_key[], npy_intp key_length)
{
    /* was signed in the original code. RDH 12/16/2002 */
    npy_intp i = 1;
    npy_intp j = 0;
    unsigned long *mt = self->key;
    npy_intp k;

    init_genrand(self, 19650218UL);
    k = (RK_STATE_LEN > key_length ? RK_STATE_LEN : key_length);
    for (; k; k--) {
        /* non linear */
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL))
            + init_key[j] + j;
        /* for > 32 bit machines */
        mt[i] &= 0xffffffffUL;
        i++;
        j++;
        if (i >= RK_STATE_LEN) {
            mt[0] = mt[RK_STATE_LEN - 1];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = RK_STATE_LEN - 1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
             - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i >= RK_STATE_LEN) {
            mt[0] = mt[RK_STATE_LEN - 1];
            i = 1;
        }
    }

    mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
    self->gauss = 0;
    self->has_gauss = 0;
    self->has_binomial = 0;
}
