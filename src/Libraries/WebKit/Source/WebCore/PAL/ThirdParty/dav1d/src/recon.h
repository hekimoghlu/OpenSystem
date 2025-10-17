/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef DAV1D_SRC_RECON_H
#define DAV1D_SRC_RECON_H

#include "src/internal.h"
#include "src/levels.h"

#define DEBUG_BLOCK_INFO 0 && \
        f->frame_hdr->frame_offset == 2 && t->by >= 0 && t->by < 4 && \
        t->bx >= 8 && t->bx < 12
#define DEBUG_B_PIXELS 0

#define decl_recon_b_intra_fn(name) \
void (name)(Dav1dTaskContext *t, enum BlockSize bs, \
            enum EdgeFlags intra_edge_flags, const Av1Block *b)
typedef decl_recon_b_intra_fn(*recon_b_intra_fn);

#define decl_recon_b_inter_fn(name) \
int (name)(Dav1dTaskContext *t, enum BlockSize bs, const Av1Block *b)
typedef decl_recon_b_inter_fn(*recon_b_inter_fn);

#define decl_filter_sbrow_fn(name) \
void (name)(Dav1dFrameContext *f, int sby)
typedef decl_filter_sbrow_fn(*filter_sbrow_fn);

#define decl_backup_ipred_edge_fn(name) \
void (name)(Dav1dTaskContext *t)
typedef decl_backup_ipred_edge_fn(*backup_ipred_edge_fn);

#define decl_read_coef_blocks_fn(name) \
void (name)(Dav1dTaskContext *t, enum BlockSize bs, const Av1Block *b)
typedef decl_read_coef_blocks_fn(*read_coef_blocks_fn);

decl_recon_b_intra_fn(dav1d_recon_b_intra_8bpc);
decl_recon_b_intra_fn(dav1d_recon_b_intra_16bpc);

decl_recon_b_inter_fn(dav1d_recon_b_inter_8bpc);
decl_recon_b_inter_fn(dav1d_recon_b_inter_16bpc);

decl_filter_sbrow_fn(dav1d_filter_sbrow_8bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_16bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_deblock_cols_8bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_deblock_cols_16bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_deblock_rows_8bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_deblock_rows_16bpc);
void dav1d_filter_sbrow_cdef_8bpc(Dav1dTaskContext *tc, int sby);
void dav1d_filter_sbrow_cdef_16bpc(Dav1dTaskContext *tc, int sby);
decl_filter_sbrow_fn(dav1d_filter_sbrow_resize_8bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_resize_16bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_lr_8bpc);
decl_filter_sbrow_fn(dav1d_filter_sbrow_lr_16bpc);

decl_backup_ipred_edge_fn(dav1d_backup_ipred_edge_8bpc);
decl_backup_ipred_edge_fn(dav1d_backup_ipred_edge_16bpc);

decl_read_coef_blocks_fn(dav1d_read_coef_blocks_8bpc);
decl_read_coef_blocks_fn(dav1d_read_coef_blocks_16bpc);

#endif /* DAV1D_SRC_RECON_H */
