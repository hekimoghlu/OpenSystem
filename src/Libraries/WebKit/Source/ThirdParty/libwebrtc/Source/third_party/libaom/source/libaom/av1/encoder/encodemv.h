/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
#ifndef AOM_AV1_ENCODER_ENCODEMV_H_
#define AOM_AV1_ENCODER_ENCODEMV_H_

#include "av1/encoder/encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void av1_encode_mv(AV1_COMP *cpi, aom_writer *w, ThreadData *td, const MV *mv,
                   const MV *ref, nmv_context *mvctx, int usehp);

void av1_update_mv_stats(const MV *mv, const MV *ref, nmv_context *mvctx,
                         MvSubpelPrecision precision);

void av1_build_nmv_cost_table(int *mvjoint, int *mvcost[2],
                              const nmv_context *mvctx,
                              MvSubpelPrecision precision);
void av1_build_nmv_component_cost_table(int *mvcost,
                                        const nmv_component *const mvcomp,
                                        MvSubpelPrecision precision);

void av1_update_mv_count(ThreadData *td);

void av1_encode_dv(aom_writer *w, const MV *mv, const MV *ref,
                   nmv_context *mvctx);
int_mv av1_get_ref_mv(const MACROBLOCK *x, int ref_idx);
int_mv av1_get_ref_mv_from_stack(int ref_idx,
                                 const MV_REFERENCE_FRAME *ref_frame,
                                 int ref_mv_idx,
                                 const MB_MODE_INFO_EXT *mbmi_ext);
void av1_find_best_ref_mvs_from_stack(int allow_hp,
                                      const MB_MODE_INFO_EXT *mbmi_ext,
                                      MV_REFERENCE_FRAME ref_frame,
                                      int_mv *nearest_mv, int_mv *near_mv,
                                      int is_integer);

static inline MV_JOINT_TYPE av1_get_mv_joint(const MV *mv) {
  // row:  Z  col:  Z  | MV_JOINT_ZERO   (0)
  // row:  Z  col: NZ  | MV_JOINT_HNZVZ  (1)
  // row: NZ  col:  Z  | MV_JOINT_HZVNZ  (2)
  // row: NZ  col: NZ  | MV_JOINT_HNZVNZ (3)
  return (!!mv->col) | ((!!mv->row) << 1);
}

static inline int av1_mv_class_base(MV_CLASS_TYPE c) {
  return c ? CLASS0_SIZE << (c + 2) : 0;
}

// If n != 0, returns the floor of log base 2 of n. If n == 0, returns 0.
static inline uint8_t av1_log_in_base_2(unsigned int n) {
  // get_msb() is only valid when n != 0.
  return n == 0 ? 0 : get_msb(n);
}

static inline MV_CLASS_TYPE av1_get_mv_class(int z, int *offset) {
  assert(z >= 0);
  const MV_CLASS_TYPE c = (MV_CLASS_TYPE)av1_log_in_base_2(z >> 3);
  assert(c <= MV_CLASS_10);
  if (offset) *offset = z - av1_mv_class_base(c);
  return c;
}

static inline int av1_check_newmv_joint_nonzero(const AV1_COMMON *cm,
                                                MACROBLOCK *const x) {
  (void)cm;
  MACROBLOCKD *xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = xd->mi[0];
  const PREDICTION_MODE this_mode = mbmi->mode;
  if (this_mode == NEW_NEWMV) {
    const int_mv ref_mv_0 = av1_get_ref_mv(x, 0);
    const int_mv ref_mv_1 = av1_get_ref_mv(x, 1);
    if (mbmi->mv[0].as_int == ref_mv_0.as_int ||
        mbmi->mv[1].as_int == ref_mv_1.as_int) {
      return 0;
    }
  } else if (this_mode == NEAREST_NEWMV || this_mode == NEAR_NEWMV) {
    const int_mv ref_mv_1 = av1_get_ref_mv(x, 1);
    if (mbmi->mv[1].as_int == ref_mv_1.as_int) {
      return 0;
    }
  } else if (this_mode == NEW_NEARESTMV || this_mode == NEW_NEARMV) {
    const int_mv ref_mv_0 = av1_get_ref_mv(x, 0);
    if (mbmi->mv[0].as_int == ref_mv_0.as_int) {
      return 0;
    }
  } else if (this_mode == NEWMV) {
    const int_mv ref_mv_0 = av1_get_ref_mv(x, 0);
    if (mbmi->mv[0].as_int == ref_mv_0.as_int) {
      return 0;
    }
  }
  return 1;
}
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_ENCODEMV_H_
