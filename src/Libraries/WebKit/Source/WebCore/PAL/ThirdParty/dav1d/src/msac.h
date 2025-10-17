/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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
#ifndef DAV1D_SRC_MSAC_H
#define DAV1D_SRC_MSAC_H

#include <stdint.h>
#include <stdlib.h>

#include "common/intops.h"

typedef size_t ec_win;

typedef struct MsacContext {
    const uint8_t *buf_pos;
    const uint8_t *buf_end;
    ec_win dif;
    unsigned rng;
    int cnt;
    int allow_update_cdf;

#if ARCH_X86_64 && HAVE_ASM
    unsigned (*symbol_adapt16)(struct MsacContext *s, uint16_t *cdf, size_t n_symbols);
#endif
} MsacContext;

#if HAVE_ASM
#if ARCH_AARCH64 || ARCH_ARM
#include "src/arm/msac.h"
#elif ARCH_X86
#include "src/x86/msac.h"
#endif
#endif

void dav1d_msac_init(MsacContext *s, const uint8_t *data, size_t sz,
                     int disable_cdf_update_flag);
unsigned dav1d_msac_decode_symbol_adapt_c(MsacContext *s, uint16_t *cdf,
                                          size_t n_symbols);
unsigned dav1d_msac_decode_bool_adapt_c(MsacContext *s, uint16_t *cdf);
unsigned dav1d_msac_decode_bool_equi_c(MsacContext *s);
unsigned dav1d_msac_decode_bool_c(MsacContext *s, unsigned f);
unsigned dav1d_msac_decode_hi_tok_c(MsacContext *s, uint16_t *cdf);
int dav1d_msac_decode_subexp(MsacContext *s, int ref, int n, unsigned k);

/* Supported n_symbols ranges: adapt4: 1-4, adapt8: 1-7, adapt16: 3-15 */
#ifndef dav1d_msac_decode_symbol_adapt4
#define dav1d_msac_decode_symbol_adapt4  dav1d_msac_decode_symbol_adapt_c
#endif
#ifndef dav1d_msac_decode_symbol_adapt8
#define dav1d_msac_decode_symbol_adapt8  dav1d_msac_decode_symbol_adapt_c
#endif
#ifndef dav1d_msac_decode_symbol_adapt16
#define dav1d_msac_decode_symbol_adapt16 dav1d_msac_decode_symbol_adapt_c
#endif
#ifndef dav1d_msac_decode_bool_adapt
#define dav1d_msac_decode_bool_adapt     dav1d_msac_decode_bool_adapt_c
#endif
#ifndef dav1d_msac_decode_bool_equi
#define dav1d_msac_decode_bool_equi      dav1d_msac_decode_bool_equi_c
#endif
#ifndef dav1d_msac_decode_bool
#define dav1d_msac_decode_bool           dav1d_msac_decode_bool_c
#endif
#ifndef dav1d_msac_decode_hi_tok
#define dav1d_msac_decode_hi_tok         dav1d_msac_decode_hi_tok_c
#endif

static inline unsigned dav1d_msac_decode_bools(MsacContext *const s, unsigned n) {
    unsigned v = 0;
    while (n--)
        v = (v << 1) | dav1d_msac_decode_bool_equi(s);
    return v;
}

static inline int dav1d_msac_decode_uniform(MsacContext *const s, const unsigned n) {
    assert(n > 0);
    const int l = ulog2(n) + 1;
    assert(l > 1);
    const unsigned m = (1 << l) - n;
    const unsigned v = dav1d_msac_decode_bools(s, l - 1);
    return v < m ? v : (v << 1) - m + dav1d_msac_decode_bool_equi(s);
}

#endif /* DAV1D_SRC_MSAC_H */
