/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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
#ifndef DAV1D_SRC_ITX_H
#define DAV1D_SRC_ITX_H

#include <stddef.h>

#include "common/bitdepth.h"

#include "src/levels.h"

#define decl_itx_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, coef *coeff, int eob \
            HIGHBD_DECL_SUFFIX)
typedef decl_itx_fn(*itxfm_fn);

typedef struct Dav1dInvTxfmDSPContext {
    itxfm_fn itxfm_add[N_RECT_TX_SIZES][N_TX_TYPES_PLUS_LL];
} Dav1dInvTxfmDSPContext;

bitfn_decls(void dav1d_itx_dsp_init, Dav1dInvTxfmDSPContext *c, int bpc);
bitfn_decls(void dav1d_itx_dsp_init_arm, Dav1dInvTxfmDSPContext *c, int bpc);
bitfn_decls(void dav1d_itx_dsp_init_x86, Dav1dInvTxfmDSPContext *c, int bpc);

#endif /* DAV1D_SRC_ITX_H */
