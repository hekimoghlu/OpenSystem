/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 15, 2022.
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
#ifndef AOM_AV1_ENCODER_PASS2_STRATEGY_H_
#define AOM_AV1_ENCODER_PASS2_STRATEGY_H_

#ifdef __cplusplus
extern "C" {
#endif

struct AV1_COMP;
struct EncodeFrameParams;

#include "av1/encoder/encoder.h"

/*!
 * \brief accumulated stats and features in a gf group
 */
typedef struct {
  /*!\cond */
  double gf_group_err;
  double gf_group_raw_error;
  double gf_group_skip_pct;
  double gf_group_inactive_zone_rows;

  double mv_ratio_accumulator;
  double decay_accumulator;
  double zero_motion_accumulator;
  double loop_decay_rate;
  double last_loop_decay_rate;
  double this_frame_mv_in_out;
  double mv_in_out_accumulator;
  double abs_mv_in_out_accumulator;

  double avg_sr_coded_error;
  double avg_pcnt_second_ref;
  double avg_new_mv_count;
  double avg_wavelet_energy;
  double avg_raw_err_stdev;
  int non_zero_stdev_count;
  /*!\endcond */
} GF_GROUP_STATS;

/*!
 * \brief accumulated stats and features for a frame
 */
typedef struct {
  /*!\cond */
  double frame_err;
  double frame_coded_error;
  double frame_sr_coded_error;
  /*!\endcond */
} GF_FRAME_STATS;
/*!\cond */

void av1_init_second_pass(struct AV1_COMP *cpi);

void av1_init_single_pass_lap(AV1_COMP *cpi);

/*!\endcond */
/*!\brief Main per frame entry point for second pass of two pass encode
 *
 *\ingroup rate_control
 *
 * This function is called for each frame in the second pass of a two pass
 * encode. It checks the frame type and if a new KF or GF/ARF is due.
 * When a KF is due it calls find_next_key_frame() to work out how long
 * this key frame group will be and assign bits to the key frame.
 * At the start of a new GF/ARF group it calls calculate_gf_length()
 * and define_gf_group() which are the main functions responsible for
 * defining the size and structure of the new GF/ARF group.
 *
 * \param[in]    cpi           Top - level encoder instance structure
 * \param[in]    frame_params  Per frame encoding parameters
 * \param[in]    frame_flags   Frame type and coding flags
 *
 * \remark No return but analyses first pass stats and assigns a target
 *         number of bits to the current frame and a target Q range.
 */
void av1_get_second_pass_params(struct AV1_COMP *cpi,
                                struct EncodeFrameParams *const frame_params,
                                unsigned int frame_flags);

/*!\brief Adjustments to two pass and rate control after each frame.
 *
 *\ingroup rate_control
 *
 * This function is called after each frame to make adjustments to
 * heuristics and data structures that relate to rate control.
 *
 * \param[in]    cpi       Top - level encoder instance structure
 *
 * \remark No return value but this function updates various rate control
 *         related data structures that for example track overshoot and
 *         undershoot.
 */
void av1_twopass_postencode_update(struct AV1_COMP *cpi);

int av1_calc_arf_boost(const TWO_PASS *twopass,
                       const TWO_PASS_FRAME *twopass_frame,
                       const PRIMARY_RATE_CONTROL *p_rc, FRAME_INFO *frame_info,
                       int offset, int f_frames, int b_frames,
                       int *num_fpstats_used, int *num_fpstats_required,
                       int project_gfu_boost);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_PASS2_STRATEGY_H_
