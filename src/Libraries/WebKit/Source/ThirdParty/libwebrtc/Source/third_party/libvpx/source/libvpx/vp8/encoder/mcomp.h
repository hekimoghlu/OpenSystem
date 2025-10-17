/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 11, 2023.
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
#ifndef VPX_VP8_ENCODER_MCOMP_H_
#define VPX_VP8_ENCODER_MCOMP_H_

#include "block.h"
#include "vpx_dsp/variance.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The maximum number of steps in a step search given the largest allowed
 * initial step
 */
#define MAX_MVSEARCH_STEPS 8

/* Max full pel mv specified in 1 pel units */
#define MAX_FULL_PEL_VAL ((1 << (MAX_MVSEARCH_STEPS)) - 1)

/* Maximum size of the first step in full pel units */
#define MAX_FIRST_STEP (1 << (MAX_MVSEARCH_STEPS - 1))

int vp8_mv_bit_cost(int_mv *mv, int_mv *ref, int *mvcost[2], int Weight);
void vp8_init_dsmotion_compensation(MACROBLOCK *x, int stride);
void vp8_init3smotion_compensation(MACROBLOCK *x, int stride);

int vp8_hex_search(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                   int_mv *best_mv, int search_param, int sad_per_bit,
                   const vp8_variance_fn_ptr_t *vfp, int *mvsadcost[2],
                   int_mv *center_mv);

typedef int(fractional_mv_step_fp)(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                   int_mv *bestmv, int_mv *ref_mv,
                                   int error_per_bit,
                                   const vp8_variance_fn_ptr_t *vfp,
                                   int *mvcost[2], int *distortion,
                                   unsigned int *sse);

fractional_mv_step_fp vp8_find_best_sub_pixel_step_iteratively;
fractional_mv_step_fp vp8_find_best_sub_pixel_step;
fractional_mv_step_fp vp8_find_best_half_pixel_step;
fractional_mv_step_fp vp8_skip_fractional_mv_step;

int vp8_full_search_sad(MACROBLOCK *x, BLOCK *b, BLOCKD *d, int_mv *ref_mv,
                        int sad_per_bit, int distance,
                        vp8_variance_fn_ptr_t *fn_ptr, int *mvcost[2],
                        int_mv *center_mv);

typedef int (*vp8_refining_search_fn_t)(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                        int_mv *ref_mv, int sad_per_bit,
                                        int distance,
                                        vp8_variance_fn_ptr_t *fn_ptr,
                                        int *mvcost[2], int_mv *center_mv);

typedef int (*vp8_diamond_search_fn_t)(MACROBLOCK *x, BLOCK *b, BLOCKD *d,
                                       int_mv *ref_mv, int_mv *best_mv,
                                       int search_param, int sad_per_bit,
                                       int *num00,
                                       vp8_variance_fn_ptr_t *fn_ptr,
                                       int *mvcost[2], int_mv *center_mv);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_MCOMP_H_
