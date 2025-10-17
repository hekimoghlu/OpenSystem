/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#include <string.h>

#include "vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vp8_rtcd.h"
#include "blockd.h"
#include "reconintra4x4.h"
#include "vp8/common/common.h"
#include "vpx_ports/compiler_attributes.h"

typedef void (*intra_pred_fn)(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left);

static intra_pred_fn pred[10];

void vp8_init_intra4x4_predictors_internal(void) {
  pred[B_DC_PRED] = vpx_dc_predictor_4x4;
  pred[B_TM_PRED] = vpx_tm_predictor_4x4;
  pred[B_VE_PRED] = vpx_ve_predictor_4x4;
  pred[B_HE_PRED] = vpx_he_predictor_4x4;
  pred[B_LD_PRED] = vpx_d45e_predictor_4x4;
  pred[B_RD_PRED] = vpx_d135_predictor_4x4;
  pred[B_VR_PRED] = vpx_d117_predictor_4x4;
  pred[B_VL_PRED] = vpx_d63e_predictor_4x4;
  pred[B_HD_PRED] = vpx_d153_predictor_4x4;
  pred[B_HU_PRED] = vpx_d207_predictor_4x4;
}

void vp8_intra4x4_predict(unsigned char *above, unsigned char *yleft,
                          int left_stride, B_PREDICTION_MODE b_mode,
                          unsigned char *dst, int dst_stride,
                          unsigned char top_left) {
/* Power PC implementation uses "vec_vsx_ld" to read 16 bytes from
   Above (aka, Aboveb + 4). Play it safe by reserving enough stack
   space here. Similary for "Left". */
#if HAVE_VSX
  unsigned char Aboveb[20];
#else
  unsigned char Aboveb[12];
#endif
  unsigned char *Above = Aboveb + 4;
#if HAVE_NEON
  // Neon intrinsics are unable to load 32 bits, or 4 8 bit values. Instead, it
  // over reads but does not use the extra 4 values.
  unsigned char Left[8];
#if VPX_WITH_ASAN
  // Silence an 'uninitialized read' warning. Although uninitialized values are
  // indeed read, they are not used.
  vp8_zero_array(Left, 8);
#endif  // VPX_WITH_ASAN
#elif HAVE_VSX
  unsigned char Left[16];
#else
  unsigned char Left[4];
#endif  // HAVE_NEON

  Left[0] = yleft[0];
  Left[1] = yleft[left_stride];
  Left[2] = yleft[2 * left_stride];
  Left[3] = yleft[3 * left_stride];
  memcpy(Above, above, 8);
  Above[-1] = top_left;

  pred[b_mode](dst, dst_stride, Above, Left);
}
