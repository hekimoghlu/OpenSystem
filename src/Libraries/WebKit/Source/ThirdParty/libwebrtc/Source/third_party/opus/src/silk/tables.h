/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 11, 2023.
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
#ifndef SILK_TABLES_H
#define SILK_TABLES_H

#include "define.h"
#include "structs.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* Entropy coding tables (with size in bytes indicated) */
extern const opus_uint8  silk_gain_iCDF[ 3 ][ N_LEVELS_QGAIN / 8 ];                                 /* 24 */
extern const opus_uint8  silk_delta_gain_iCDF[ MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1 ];   /* 41 */

extern const opus_uint8  silk_pitch_lag_iCDF[ 2 * ( PITCH_EST_MAX_LAG_MS - PITCH_EST_MIN_LAG_MS ) ];/* 32 */
extern const opus_uint8  silk_pitch_delta_iCDF[ 21 ];                                               /*  21 */
extern const opus_uint8  silk_pitch_contour_iCDF[ 34 ];                                             /*  34 */
extern const opus_uint8  silk_pitch_contour_NB_iCDF[ 11 ];                                          /*  11 */
extern const opus_uint8  silk_pitch_contour_10_ms_iCDF[ 12 ];                                       /*  12 */
extern const opus_uint8  silk_pitch_contour_10_ms_NB_iCDF[ 3 ];                                     /*   3 */

extern const opus_uint8  silk_pulses_per_block_iCDF[ N_RATE_LEVELS ][ SILK_MAX_PULSES + 2 ];        /* 180 */
extern const opus_uint8  silk_pulses_per_block_BITS_Q5[ N_RATE_LEVELS - 1 ][ SILK_MAX_PULSES + 2 ]; /* 162 */

extern const opus_uint8  silk_rate_levels_iCDF[ 2 ][ N_RATE_LEVELS - 1 ];                           /*  18 */
extern const opus_uint8  silk_rate_levels_BITS_Q5[ 2 ][ N_RATE_LEVELS - 1 ];                        /*  18 */

extern const opus_uint8  silk_max_pulses_table[ 4 ];                                                /*   4 */

extern const opus_uint8  silk_shell_code_table0[ 152 ];                                             /* 152 */
extern const opus_uint8  silk_shell_code_table1[ 152 ];                                             /* 152 */
extern const opus_uint8  silk_shell_code_table2[ 152 ];                                             /* 152 */
extern const opus_uint8  silk_shell_code_table3[ 152 ];                                             /* 152 */
extern const opus_uint8  silk_shell_code_table_offsets[ SILK_MAX_PULSES + 1 ];                      /*  17 */

extern const opus_uint8  silk_lsb_iCDF[ 2 ];                                                        /*   2 */

extern const opus_uint8  silk_sign_iCDF[ 42 ];                                                      /*  42 */

extern const opus_uint8  silk_uniform3_iCDF[ 3 ];                                                   /*   3 */
extern const opus_uint8  silk_uniform4_iCDF[ 4 ];                                                   /*   4 */
extern const opus_uint8  silk_uniform5_iCDF[ 5 ];                                                   /*   5 */
extern const opus_uint8  silk_uniform6_iCDF[ 6 ];                                                   /*   6 */
extern const opus_uint8  silk_uniform8_iCDF[ 8 ];                                                   /*   8 */

extern const opus_uint8  silk_NLSF_EXT_iCDF[ 7 ];                                                   /*   7 */

extern const opus_uint8  silk_LTP_per_index_iCDF[ 3 ];                                              /*   3 */
extern const opus_uint8  * const silk_LTP_gain_iCDF_ptrs[ NB_LTP_CBKS ];                            /*   3 */
extern const opus_uint8  * const silk_LTP_gain_BITS_Q5_ptrs[ NB_LTP_CBKS ];                         /*   3 */
extern const opus_int8   * const silk_LTP_vq_ptrs_Q7[ NB_LTP_CBKS ];                                /* 168 */
extern const opus_uint8  * const silk_LTP_vq_gain_ptrs_Q7[NB_LTP_CBKS];
extern const opus_int8   silk_LTP_vq_sizes[ NB_LTP_CBKS ];                                          /*   3 */

extern const opus_uint8  silk_LTPscale_iCDF[ 3 ];                                                   /*   4 */
extern const opus_int16  silk_LTPScales_table_Q14[ 3 ];                                             /*   6 */

extern const opus_uint8  silk_type_offset_VAD_iCDF[ 4 ];                                            /*   4 */
extern const opus_uint8  silk_type_offset_no_VAD_iCDF[ 2 ];                                         /*   2 */

extern const opus_int16  silk_stereo_pred_quant_Q13[ STEREO_QUANT_TAB_SIZE ];                       /*  32 */
extern const opus_uint8  silk_stereo_pred_joint_iCDF[ 25 ];                                         /*  25 */
extern const opus_uint8  silk_stereo_only_code_mid_iCDF[ 2 ];                                       /*   2 */

extern const opus_uint8  * const silk_LBRR_flags_iCDF_ptr[ 2 ];                                     /*  10 */

extern const opus_uint8  silk_NLSF_interpolation_factor_iCDF[ 5 ];                                  /*   5 */

extern const silk_NLSF_CB_struct silk_NLSF_CB_WB;                                                   /* 1040 */
extern const silk_NLSF_CB_struct silk_NLSF_CB_NB_MB;                                                /* 728 */

/* Quantization offsets */
extern const opus_int16  silk_Quantization_Offsets_Q10[ 2 ][ 2 ];                                   /*   8 */

/* Interpolation points for filter coefficients used in the bandwidth transition smoother */
extern const opus_int32  silk_Transition_LP_B_Q28[ TRANSITION_INT_NUM ][ TRANSITION_NB ];           /*  60 */
extern const opus_int32  silk_Transition_LP_A_Q28[ TRANSITION_INT_NUM ][ TRANSITION_NA ];           /*  60 */

/* Rom table with cosine values */
extern const opus_int16  silk_LSFCosTab_FIX_Q12[ LSF_COS_TAB_SZ_FIX + 1 ];                          /* 258 */

#ifdef __cplusplus
}
#endif

#endif
